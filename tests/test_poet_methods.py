"""Geometry semantics and model/gradient contracts; no dataset-score claims."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import numpy as np
from poet_geometry import GEOMETRY_DIM, build_geometry_features, geometry_spec


def raw_fixture(frames=3):
    raw = {name: np.zeros((frames, points, 3), dtype=np.float64)
           for name, points in {"body_pose": 33, "face_mesh": 468, "right_hand": 21, "left_hand": 21}.items()}
    raw.update({name + "_present": np.ones(frames, dtype=np.bool_) for name in tuple(raw)})
    raw["body_visibility"] = np.ones((frames, 33), dtype=np.float64)
    raw["body_pose"][:, 11, :2] = [0.3, 0.6]
    raw["body_pose"][:, 12, :2] = [0.7, 0.6]
    raw["face_mesh"][:, 1, :2] = [0.5, 0.2]
    raw["face_mesh"][:, 13, :2] = [0.5, 0.25]
    raw["face_mesh"][:, 14, :2] = [0.5, 0.35]
    raw["right_hand"][:, 0, :2] = [0.1, 0.4]
    raw["left_hand"][:, 0, :2] = [0.9, 0.4]
    return raw


class GeometryTests(unittest.TestCase):
    def test_known_cross_part_geometry_and_direction(self):
        result = build_geometry_features(raw_fixture(), np.arange(3))
        self.assertEqual(result.shape, (3, 28))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[0, :4], [1, -0.5, np.sqrt(1.25), 1], rtol=1e-6)
        np.testing.assert_allclose(result[0, 4:8], [1, -0.25, np.sqrt(1.0625), 1], rtol=1e-6)
        np.testing.assert_allclose(result[0, 24:28], [2, 0, 2, 1], rtol=1e-6)

    def test_temporal_duplicates_crops_and_zero_padding_use_supplied_indices(self):
        raw = raw_fixture()
        raw["right_hand"][1, 0, 0] += 0.1
        full = build_geometry_features(raw, np.arange(3))
        selected = build_geometry_features(raw, np.array([2, 1, 1, -1, 0]))
        np.testing.assert_array_equal(selected[:3], full[[2, 1, 1]])
        np.testing.assert_array_equal(selected[3], np.zeros(GEOMETRY_DIM))
        np.testing.assert_array_equal(selected[4], full[0])

    def test_translation_uniform_scale_and_part_local_z_do_not_change_geometry(self):
        raw = raw_fixture()
        expected = build_geometry_features(raw, np.arange(3))
        changed = copy.deepcopy(raw)
        for name in ("body_pose", "face_mesh", "right_hand", "left_hand"):
            changed[name][..., :2] = changed[name][..., :2] * 2.3 + [0.17, -0.09]
            changed[name][..., 2] = np.nan  # z is deliberately outside this feature definition.
        np.testing.assert_allclose(build_geometry_features(changed, np.arange(3)), expected, rtol=1e-6, atol=1e-6)

    def test_absent_parts_do_not_make_fake_distances(self):
        raw = raw_fixture()
        raw["right_hand_present"][1] = False
        # Stale but finite coordinates must still be ignored according to the mask.
        raw["right_hand"][1] = 123.0
        result = build_geometry_features(raw, np.arange(3)).reshape(3, 7, 4)
        np.testing.assert_array_equal(result[1, [0, 1, 2, 6]], 0)
        self.assertTrue((result[1, [3, 4, 5], 3] == 1).all())

    def test_zero_image_coordinate_is_not_a_missing_flag(self):
        raw = raw_fixture()
        raw["face_mesh"][:, 1, :2] = 0
        result = build_geometry_features(raw, np.arange(3))
        self.assertEqual(result[0, 3], 1)
        self.assertGreater(result[0, 2], 0)

    def test_scale_uses_only_visible_shoulders_and_masks_current_body_anchor(self):
        raw = raw_fixture()
        raw["body_visibility"][0, 11] = 0.499
        raw["body_pose"][0, 12, 0] = 99
        result = build_geometry_features(raw, np.arange(3)).reshape(3, 7, 4)
        np.testing.assert_array_equal(result[0, [2, 5]], 0)
        self.assertAlmostEqual(result[0, 0, 0], 1.0)
        self.assertEqual(result[1, 2, 3], 1)

    def test_missing_or_degenerate_scale_disables_geometry_without_fallback(self):
        for mode in ("missing", "low_visibility", "zero_width"):
            raw = raw_fixture()
            if mode == "missing":
                raw["body_pose_present"][:] = False
            elif mode == "low_visibility":
                raw["body_visibility"][:, 12] = 0.1
            else:
                raw["body_pose"][:, 12] = raw["body_pose"][:, 11]
            np.testing.assert_array_equal(build_geometry_features(raw, np.arange(3)), 0)

    def test_bad_indices_masks_and_nonfinite_xy_are_rejected(self):
        for indices in (np.array([1.0]), np.array([-2]), np.array([3]), np.array([], dtype=int)):
            with self.assertRaises(ValueError):
                build_geometry_features(raw_fixture(), indices)
        raw = raw_fixture()
        raw["face_mesh_present"] = np.ones(3, dtype=int)
        with self.assertRaisesRegex(ValueError, "boolean"):
            build_geometry_features(raw, np.arange(3))
        raw = raw_fixture()
        raw["face_mesh"][0, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            build_geometry_features(raw, np.arange(3))

    def test_spec_is_independent_plain_metadata(self):
        changed = geometry_spec()
        changed["anchors"]["nose"]["indices"][0] = 99
        self.assertEqual(geometry_spec()["anchors"]["nose"]["indices"], [1])


HAS_MODEL_RUNTIME = (all(importlib.util.find_spec(name) is not None for name in ("torch", "mediapipe"))
                     and (SCRIPTS.parent / "vendor/poet/model.py").is_file())


@unittest.skipUnless(HAS_MODEL_RUNTIME, "PoET model checks require the optional torch/MediaPipe runtime and downloaded vendor code")
class MethodModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import poet_methods
        from _poet_runtime import Transformer_CLS
        cls.torch = torch
        cls.methods = poet_methods
        cls.model_class = Transformer_CLS
        torch.set_num_threads(1)

    def base(self, training=False):
        self.torch.manual_seed(17)
        model = self.model_class(input_size=188, num_classes=7, max_seq_length=8,
                                 num_encoder_layers=2, num_attention_heads=2, hidden_dropout_prob=0.2,
                                 hidden_size=16, mlp_classifier="MLP1", mlp_hidden_size=0)
        model.train(training)
        return model

    def inputs(self):
        torch = self.torch
        torch.manual_seed(21)
        x = torch.randn(2, 8, 188)
        x[:, -1] = 0
        geo = torch.from_numpy(build_geometry_features(raw_fixture(), np.array([0, 0, 1, 1, 1, 2, 2, -1])))
        return x, geo.unsqueeze(0).repeat(2, 1, 1)

    def test_all_method_initial_eval_logits_exactly_equal_unmodified_vendor(self):
        torch = self.torch
        base = self.base()
        x, geometry = self.inputs()
        with torch.inference_mode():
            expected = base(x, "cpu")
            for method in ("baseline", "geometry", "attributes"):
                wrapped = self.methods.PoETMethod(copy.deepcopy(base), method,
                                                  attribute_dims={"shape": 3} if method == "attributes" else None)
                actual = wrapped(x, "cpu", geometry=geometry if method == "geometry" else None)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless((SCRIPTS.parent / "assets/checkpoints/ASL_Citizen_PoET.pth").is_file(),
                         "Official checkpoint must be downloaded separately")
    def test_official_checkpoint_initial_logits_exactly_equal_for_all_methods(self):
        from _poet_runtime import load_bundle
        torch = self.torch
        bundle = load_bundle(SCRIPTS.parent / "assets/checkpoints", torch.device("cpu"))
        torch.manual_seed(107)
        x = torch.randn(2, bundle.preprocessing.max_seq_length, 188)
        geo = torch.from_numpy(build_geometry_features(raw_fixture(), np.zeros(x.shape[1], dtype=np.int64)))
        geo = geo.unsqueeze(0).repeat(2, 1, 1)
        with torch.inference_mode():
            expected = bundle.model(x, "cpu")
            for method in ("baseline", "geometry", "attributes"):
                model = self.methods.PoETMethod(copy.deepcopy(bundle.model), method,
                                                attribute_dims={"synthetic_contract": 3} if method == "attributes" else None)
                torch.testing.assert_close(model(x, geometry=geo if method == "geometry" else None), expected, rtol=0, atol=0)

    def test_geometry_aligns_with_actual_official_train_and_eval_temporal_indices(self):
        from _poet_runtime import prepare_features
        args = SimpleNamespace(keypoints=188, cut=1, fix_lenght=1, padding="copy",
                               troncate="downsampling", max_seq_length=128)
        for frames in (30, 300):
            raw = raw_fixture(frames)
            raw["right_hand"][:, 0, 0] = np.linspace(0.1, 0.4, frames)
            features = np.ones((frames, 188))
            features[:, 0] = np.arange(frames)
            features[:2, -80:] = 0
            features[-2:, -80:] = 0
            full_geo = build_geometry_features(raw, np.arange(frames))
            for mode in ("train", "eval"):
                np.random.seed(27)
                main, indices = prepare_features(features, args, mode=mode)
                np.testing.assert_array_equal(main[:, 0].numpy(), indices)
                aligned = build_geometry_features(raw, indices)
                self.assertEqual(aligned.shape, (len(main), 28))
                np.testing.assert_array_equal(aligned, full_geo[indices])

    def test_private_head_initialization_preserves_torch_rng(self):
        torch = self.torch
        for method in ("geometry", "attributes"):
            base = self.base()
            before = torch.get_rng_state().clone()
            self.methods.PoETMethod(base, method, attribute_dims={"shape": 3} if method == "attributes" else None)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_train_logits_and_base_gradients_equal_vendor_with_identical_dropout_rng(self):
        torch = self.torch
        for method in ("baseline", "geometry", "attributes"):
            base = self.base(training=True)
            wrapped = self.methods.PoETMethod(copy.deepcopy(base), method,
                                              attribute_dims={"shape": 3} if method == "attributes" else None)
            x, geometry = self.inputs()
            torch.manual_seed(54)
            expected = base(x, "cpu")
            expected.square().sum().backward()
            torch.manual_seed(54)
            actual = wrapped(x, "cpu", geometry=geometry if method == "geometry" else None)
            actual.square().sum().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for (_, first), (_, second) in zip(base.named_parameters(), wrapped.base.named_parameters()):
                torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)

    def test_geometry_last_layer_learns_first_then_hidden_layer_receives_gradient(self):
        torch = self.torch
        wrapped = self.methods.PoETMethod(self.base(), "geometry")
        x, geometry = self.inputs()
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        wrapped(x, geometry=geometry).square().sum().backward()
        self.assertGreater(wrapped.geometry_branch[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(wrapped.geometry_branch[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        wrapped(x, geometry=geometry).square().sum().backward()
        self.assertGreater(wrapped.geometry_branch[0].weight.grad.abs().sum().item(), 0)

    def test_all_missing_geometry_cannot_inject_learned_branch_bias(self):
        torch = self.torch
        base = self.base()
        wrapped = self.methods.PoETMethod(copy.deepcopy(base), "geometry")
        x, geometry = self.inputs()
        with torch.no_grad():
            wrapped.geometry_branch[-1].bias.fill_(5.0)
            torch.testing.assert_close(wrapped(x, geometry=torch.zeros_like(geometry)), base(x, "cpu"), rtol=0, atol=0)

    def test_auxiliary_loss_reaches_shared_encoder_without_word_loss_or_labels_in_forward(self):
        torch = self.torch
        model = self.methods.PoETMethod(self.base(), "attributes", attribute_dims={"shape": 3, "location": 2})
        x, _ = self.inputs()
        word_logits, attribute_logits = model.forward_with_attributes(x)
        torch.testing.assert_close(word_logits, model(x), rtol=0, atol=0)
        targets = {name: torch.zeros_like(value) for name, value in attribute_logits.items()}
        masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in attribute_logits.items()}
        loss = self.methods.masked_attribute_bce(attribute_logits, targets, masks)
        loss.backward()
        self.assertGreater(model.base.embedding[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.base.transformer_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.base.mlp_classifier.L.weight.grad)

    def test_missing_auxiliary_targets_are_ignored_even_if_nan(self):
        torch = self.torch
        logits = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        target = torch.tensor([[1.0, float("nan"), float("nan")]])
        mask = torch.tensor([[True, False, False]])
        loss = self.methods.masked_attribute_bce({"a": logits}, {"a": target}, {"a": mask})
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, :1], target[0, :1])
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        self.assertNotEqual(logits.grad[0, 0].item(), 0)
        torch.testing.assert_close(logits.grad[0, 1:], torch.zeros(2), rtol=0, atol=0)

    def test_all_masked_auxiliary_loss_is_zero_with_finite_zero_gradients(self):
        torch = self.torch
        logits = torch.randn(2, 3, requires_grad=True)
        loss = self.methods.masked_attribute_bce({"a": logits}, {"a": torch.full_like(logits, float("nan"))},
                                                {"a": torch.zeros_like(logits, dtype=torch.bool)})
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)

    def test_bad_attribute_contract_or_combination_is_rejected(self):
        torch = self.torch
        with self.assertRaises(ValueError):
            self.methods.PoETMethod(self.base(), "combined")
        with self.assertRaises(ValueError):
            self.methods.PoETMethod(self.base(), "attributes")
        with self.assertRaises(ValueError):
            self.methods.PoETMethod(self.base(), "geometry", attribute_dims={"shape": 3})
        logits = torch.zeros(1, 2)
        for target, mask in ((torch.ones(1, 1), torch.ones(1, 2, dtype=torch.bool)),
                             (torch.full((1, 2), 0.5), torch.ones(1, 2, dtype=torch.bool)),
                             (torch.ones(1, 2), torch.ones(1, 2))):
            with self.assertRaises(ValueError):
                self.methods.masked_attribute_bce({"a": logits}, {"a": target}, {"a": mask})

    def test_method_state_roundtrip_and_parameter_count(self):
        torch = self.torch
        x, geometry = self.inputs()
        model = self.methods.PoETMethod(self.base(), "geometry")
        restored = self.methods.PoETMethod(self.base(), "geometry", head_seed=5)
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(model(x, geometry=geometry), restored(x, geometry=geometry), rtol=0, atol=0)
        self.assertEqual(model.protocol()["added_parameters"], 28 * 32 + 32 + 32 * 16 + 16)


if __name__ == "__main__":
    unittest.main()
