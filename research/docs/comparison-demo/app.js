(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const data = window.COMPARISON_DATA || null;
  const order = ['knn', 'linear_svm', 'random_forest', 'poet_official_v2'];
  const liveKeys = { poet_official: 'poet_official_v2' };
  const colors = { knn: '#c69249', linear_svm: '#5e84ac', random_forest: '#9b80ae', poet_official_v2: '#2d896e', poet_geometry_retrieval: '#34765b' };
  const names = { knn: 'kNN', linear_svm: 'Linear SVM', random_forest: 'Random Forest', poet_official_v2: 'PoET', poet_geometry_retrieval: 'PoET + G + R' };
  let language = 'zh';
  try { language = localStorage.getItem('asl-lab-language') === 'en' ? 'en' : 'zh'; } catch (_) { /* File URLs may restrict storage. */ }
  const zh = {};
  document.querySelectorAll('[data-i18n]').forEach((node) => { zh[node.dataset.i18n] = node.innerHTML; });
  const en = {
    skip: 'Skip to content', navOverview: 'Overview', navCases: 'Video explorer', navAnalysis: 'Error analysis', navMethods: 'Methods', offline: 'Saved results', live: 'Service connected',
    heroTitle: 'One sign.<br><span>Four ways to recognize it.</span>', heroLead: 'Start with real videos. Compare classical classifiers with PoET, and explore where each method succeeds or fails.',
    tagTask: 'Isolated-word recognition', tagSplit: 'Official signer-disjoint splits', tagValidation: 'CHARTS: VALIDATION', taskLabel: 'THE TASK', taskText: 'One presegmented ASL word video<br>→ one label from 2,731 classes', taskBoundary: 'This system does not segment continuous signing or translate sentences.',
    trainStat: 'Training videos · fit / build reference bank', valStat: 'Validation videos · all charts on this page', classStat: 'Word classes · one shared vocabulary', testStat: 'Test videos · final evaluation done Oct 12',
    resultsTitle: 'Same task. Different results.', chartDenominator: 'All 10,304 validation videos', interpretLabel: 'HOW TO READ THIS', interpretTitle: 'Representation is part of the difference.',
    interpretText: 'The classical methods share a fixed 1,020-dimensional pose and motion descriptor. PoET learns temporal representations and uses official pretrained weights. This comparison changes more than the classifier.',
    interpretLimit: 'These scores describe our implementations and budgets. They do not establish that every SVM or random forest is unsuitable for sign recognition.',
    evidenceText: 'Shared vocabulary, official splits, and complete denominators. Fit and reference-bank construction use train; validation was used for development and model selection, so these scores are optimistic. After the methods were frozen, test was evaluated once (Oct 12). Test Top-1 for the same four models: kNN 15.52%, Linear SVM 30.93%, RF 12.54%, PoET 73.56%. Stronger variants in the final report (handedness-normalized tuned SVM 42.52%, PoET with mirror TTA 74.72%) are not part of this page.',
    explorerTitle: 'Open a video. Compare the predictions.', explorerIntro: 'Explore successes, shared failures, and disagreements. Cases are selected deliberately; their success rate is not an estimate of dataset accuracy.',
    filterLabel: 'Case type', caseLabel: 'Choose a video', noVideo: 'No video is available for this case.', groundTruth: 'DATASET LABEL', runCase: 'Recognize this video again ↗', uploadVideo: 'Upload a word video', uploadNote: 'Live inference requires the demo service. Upload limit: 25 MB and 30 seconds. Use a video containing one sign.',
    predictionsTitle: 'Candidate words · Top-5', savedPredictions: 'SAVED PREDICTIONS', scoreCaution: 'Scores have different meanings across methods. Do not compare them as confidence values.',
    analysisTitle: 'An error rate is only the beginning.', analysisIntro: 'See which examples both methods recognize, and which only one gets right. Then ask what evidence could explain the difference.',
    overlapTitle: 'How much do two methods overlap?', methodA: 'Method A', methodB: 'Method B', signerTitle: 'Performance across signers', signerNote: 'These six validation signers are absent from training. Differences between groups are observations, not explanations by themselves.',
    groupsTitle: 'When do methods fail more often?', groupsNote: 'Associations with missing landmarks or clip length are not causal evidence. Group sizes are shown alongside results.',
    analysisBoundaryTitle: 'LIMITS OF THE EVIDENCE', analysisBoundary: 'Similar hand shapes, temporal patterns, or occlusion are possible explanations, not established causes. No new expert labels or causal experiments were added. Predictions are verifiable; mechanism hypotheses remain open.',
    methodsTitle: 'What does each method actually learn?', pipelineVideo: 'Word video', pipelinePose: 'MediaPipe landmarks', pipelineFixed: 'Fixed summary → kNN / SVM / RF', pipelineLearned: 'Temporal representation → PoET',
    implementationNote: 'Implementation sources: MediaPipe and PoET use existing models; SVM / RF use scikit-learn; kNN wraps standard cosine search and voting. Our work includes feature design, data and vocabulary adaptation, unified evaluation, and analysis.',
    improvementTitle: 'A small gain.<br>A controlled comparison.', improvementText: 'Relative geometry (G) adds spatial relationships between hands and body. Retrieval (R) uses training-set PoET embeddings to rerank the original Top-5.',
    gainText: 'Mean Top-1 gain over a matched<br>fine-tuned PoET baseline · three seeds · validation',
    ablationNote: 'B is the budget-matched fine-tuned baseline, not the official PoET checkpoint in the four-method comparison above. These are existing three-seed validation experiments.',
    ablationCaution: 'G+R also turns some correct predictions into errors. The figures above are validation results. On the test set, evaluated once after the methods were frozen, G+R averaged 74.18% over three seeds versus 73.45% for B (+0.74 pp); all three seeds agree in direction and both the seed and signer 95% intervals exclude zero.',
    footerScope: 'ASL Citizen · Isolated-word recognition and method comparison', downloadData: 'Download analysis data ↗', readReport: 'Read the analysis report ↗', footerNote: 'Saved predictions can be explored offline. New model inference runs only after connecting to the demo service and submitting a video.'
  };
  const messages = {
    all: ['全部案例', 'All cases'], noData: ['分析数据尚未载入。请保留 data.js 与页面在同一目录。', 'Analysis data did not load. Keep data.js alongside this page.'],
    top1Desc: ['Top-1 · 第一候选正确率', 'Top-1 · first candidate is correct'], top5Desc: ['Top-5 · 候选集合包含真实词', 'Top-5 · candidate set contains the label'],
    correct: ['正确', 'Correct'], wrong: ['错误', 'Wrong'], unknown: ['未标注', 'Unlabeled'], noPred: ['暂无此方法预测', 'No prediction available'], waiting: ['等待实时预测', 'Waiting for live prediction'],
    observed: ['已记录的事实', 'Recorded observation'], visual: ['抽帧观察', 'Frame observation'], hypothesis: ['待验证的解释', 'Hypothesis to investigate'],
    votes: ['近邻投票比例 · 不是校准概率', 'Neighbor vote fraction · not calibrated'], margin: ['OVR 决策 margin · 可为负，不是概率', 'OVR decision margin · may be negative, not probability'],
    forestProb: ['森林类别概率估计 · 未校准', 'Forest class probability estimate · uncalibrated'], softmax: ['Softmax 概率 · 未校准', 'Softmax probability · uncalibrated'], blend: ['原 Top-5 内融合分数 · 非全词表概率', 'Blend score within original Top-5 · not full-vocabulary probability'],
    scoreGeneric: ['模型排名分数', 'Model ranking score'], bothRight: ['两者都对', 'Both correct'], onlyA: ['仅 A 正确', 'Only A correct'], onlyB: ['仅 B 正确', 'Only B correct'], bothWrong: ['两者都错', 'Both incorrect'],
    sameMethod: ['请选择两个不同的方法。', 'Choose two different methods.'], oracle: ['至少一个方法正确：{n} / {total}（{pct}）。这是事后互补性上界，不是已实现的融合成绩。', 'At least one correct: {n} / {total} ({pct}). This is a retrospective complementarity bound, not a deployed ensemble score.'],
    group: ['分组', 'Group'], n: ['样本数', 'N'], top1Header: ['Top-1 正确率', 'Top-1 accuracy'], noGroups: ['暂无分组证据。', 'No grouped evidence available.'],
    method: ['方法', 'Method'], meanSD: ['Top-1（%）· 均值 ± SD', 'Top-1 (%) · mean ± SD'], limitations: ['局限与本次设定', 'Limitations and configuration'],
    queued: ['任务已提交，等待模型处理…', 'Submitted. Waiting for inference…'], running: ['正在提取关键点并识别视频…', 'Extracting landmarks and recognizing the video…'], complete: ['本次实时识别已完成。预测卡片已更新为这次推理的结果。', 'Live inference completed. Prediction cards now show results from this request.'],
    failed: ['本次识别未完成。可继续查看已保存的案例；运行细节请查看演示服务日志。', 'Inference did not complete. Saved cases remain available; see the demo service log for details.'],
    disconnected: ['当前未连接演示服务。页面仍可浏览已有视频与已保存的预测。', 'The demo service is unavailable. Videos and saved predictions remain accessible.'],
    timeout: ['等待超时，服务可能仍在处理。请先检查任务状态，再重新提交。', 'The wait timed out; the service may still be processing. Check the job before submitting again.'],
    uploading: ['正在读取并提交视频…', 'Reading and submitting the video…'], livePredictions: ['实时推理结果', 'LIVE INFERENCE'],
    uploadLabel: ['未提供真实标签', 'No ground-truth label'], uploaded: ['用户上传', 'UPLOADED'], uploadedInfo: ['上传视频没有数据集标签，因此这里只展示预测，不计算“正确 / 错误”。', 'This upload has no dataset label. Predictions are shown without correctness judgments.'],
    liveObservation: ['当前结果为新推理。原案例的观察与假设基于保存预测；切换案例可回到原始分析。', 'These are newly computed predictions. Original case observations concern saved results; select a case again to return to that analysis.'],
    sizeError: ['请选择非空且不超过 25 MB 的视频。', 'Choose a nonempty video no larger than 25 MB.'], typeError: ['请选择 MP4、MOV 或 WebM 视频。', 'Choose an MP4, MOV, or WebM video.'], durationError: ['视频超过 30 秒。请裁出一个词的完整动作后再上传。', 'The video exceeds 30 seconds. Trim it to one complete sign before uploading.'],
    videoError: ['浏览器无法读取这段视频。请尝试标准 MP4 文件。', 'The browser could not read this video. Try a standard MP4 file.'],
    seconds: ['秒', 's'], elapsed: ['本次总耗时', 'Request time'], frames: ['帧数', 'Frames'], leftHand: ['左手检测覆盖', 'Left-hand coverage'], rightHand: ['右手检测覆盖', 'Right-hand coverage'],
    noHands: ['该视频未检测到手部关键点，强制输出的词标签不应被当作可靠识别。', 'No hand landmarks were detected. A forced word prediction should not be treated as reliable recognition.'],
    coverageNote: ['检测覆盖率不是识别准确率。', 'Detection coverage is not recognition accuracy.'], otherWarnings: ['服务报告了输入质量或模型提示，请结合检测覆盖率解读结果。', 'The service reported input-quality or model warnings. Interpret results alongside landmark coverage.']
  };
  const subtitles = {
    knn: ['找相似训练例子，再投票', 'Find similar examples, then vote'],
    linear_svm: ['为每个词学习一条分界', 'Learn one boundary per word'],
    random_forest: ['多棵决策树共同判断', 'Combine many decision trees'],
    poet_official_v2: ['学习姿态随时间的变化', 'Learn how pose evolves over time']
  };
  const fallbackDescriptions = {
    knn: ['1,020 维摘要经训练集标准化与 L2 归一化，在训练库中寻找 5 个余弦近邻，按词标签等权投票。', 'Standardize the 1,020-D descriptor using training statistics, L2-normalize it, then find five cosine neighbors in the training bank and vote.'],
    linear_svm: ['使用相同的固定特征，训练 2,731 个 one-vs-rest 线性分类器。按决策 margin 对候选词排序。', 'Fit 2,731 one-vs-rest linear classifiers on the same fixed features. Rank words by their decision margins.'],
    random_forest: ['64 棵受限规模的树分别按特征阈值分支，平均叶节点的类别分布得到最终预测。', 'Each of 64 size-constrained trees branches on feature thresholds. Average leaf class distributions to make a prediction.'],
    poet_official_v2: ['将关键点序列编码为时序表示，用 Transformer 聚合视频信息，再输出整个词表的分类分数。', 'Encode landmark sequences as temporal representations, aggregate them with a Transformer, then classify across the vocabulary.']
  };
  const state = { metric: 'top1', filter: 'all', caseId: null, ready: false, busy: false, liveResult: null, uploadUrl: null, uploadName: null, jobState: null, jobMessage: null, pairA: 'linear_svm', pairB: 'poet_official_v2', signerMethod: 'poet_official_v2', groupDimension: null };
  const t = (key) => (language === 'en' ? en[key] : zh[key]) || zh[key] || en[key] || key;
  const m = (key, vars = {}) => {
    let out = (messages[key] || [key, key])[language === 'en' ? 1 : 0];
    Object.entries(vars).forEach(([name, value]) => { out = out.replaceAll(`{${name}}`, String(value)); });
    return out;
  };
  const localized = (item, key) => item?.[`${key}_${language}`] || item?.[key] || '';
  const number = (n) => Number(n).toLocaleString('en-US');
  const percent = (n, precision = 2) => n !== null && n !== undefined && Number.isFinite(Number(n)) ? `${(Number(n) * 100).toFixed(precision)}%` : '—';
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const methodId = (id) => liveKeys[id] || id;
  const getMethod = (id) => (data?.methods || []).find((x) => x.id === id);
  const currentCase = () => (data?.cases || []).find((x) => x.id === state.caseId);
  const cases = () => (data?.cases || []).filter((x) => state.filter === 'all' || x.category === state.filter);
  const text = (id, value) => { $(id).textContent = value; };
  const options = (items, selected) => items.map(([value, label]) => `<option value="${escape(value)}" ${String(value) === String(selected) ? 'selected' : ''}>${escape(label)}</option>`).join('');

  function renderBenchmark() {
    text('metric-description', m(state.metric === 'top1' ? 'top1Desc' : 'top5Desc'));
    $('benchmark-chart').innerHTML = order.map((id) => {
      const row = getMethod(id);
      if (!row) return '';
      const value = Number(row[state.metric]);
      const count = row[`${state.metric}_correct`] ?? Math.round(value * row.count);
      return `<div class="chart-row" style="--method-color:${colors[id]}"><span class="chart-method">${escape(names[id])}</span><div class="chart-track" role="img" aria-label="${escape(`${names[id]} ${state.metric}: ${percent(value)}`)}"><div class="chart-fill" style="--width:${Math.min(100, Math.max(0, value * 100))}%"></div></div><div class="chart-value"><strong>${percent(value)}</strong><small>${number(count)} / ${number(row.count || data.count)}</small></div></div>`;
    }).join('');
    document.querySelectorAll('[data-metric]').forEach((button) => { const active = button.dataset.metric === state.metric; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active)); });
  }

  function renderSelectors() {
    const unique = Array.from(new Map((data?.cases || []).map((row) => [row.category, row])).values());
    $('case-filter').innerHTML = options([['all', m('all')], ...unique.map((row) => [row.category, localized(row, 'category')])], state.filter);
    const filtered = cases();
    if (!filtered.some((row) => row.id === state.caseId)) state.caseId = filtered[0]?.id || null;
    $('case-select').innerHTML = (state.uploadName ? `<option value="" selected disabled>${escape(m('uploaded'))} · ${escape(state.uploadName)}</option>` : '') + options(filtered.map((row) => [row.id, `${row.gloss} · ${localized(row, 'category')}`]), state.uploadName ? null : state.caseId);
    text('case-position', state.uploadName ? m('uploaded') : `${Math.max(0, filtered.findIndex((row) => row.id === state.caseId) + 1)} / ${filtered.length}`);
    $('previous-case').disabled = state.busy || filtered.length < 2;
    $('next-case').disabled = state.busy || filtered.length < 2;
  }

  function renderCase(resetVideo = true) {
    const row = currentCase();
    if (!row) return;
    if (resetVideo) {
      const video = $('case-video');
      video.pause();
      video.hidden = !row.video_file;
      $('video-placeholder').hidden = !!row.video_file;
      if (row.video_file) video.src = row.video_file; else video.removeAttribute('src');
      if (row.poster_file) video.poster = row.poster_file; else video.removeAttribute('poster');
      video.load();
      $('frame-strip').innerHTML = (row.frame_files || []).map((file, index) => `<button type="button" data-frame="${(index + 1) * 25}" aria-label="${(index + 1) * 25}%"><img src="${escape(file)}" alt="${escape(row.gloss)} · ${(index + 1) * 25}%" loading="lazy"><span>${(index + 1) * 25}%</span></button>`).join('');
    }
    text('ground-truth', row.gloss);
    text('truth-caption', t('groundTruth'));
    text('case-source', 'VALIDATION');
    const observation = localized(row, 'observations');
    const hypothesis = localized(row, 'hypothesis');
    const visual = localized(row, 'visual_observation');
    $('case-observations').innerHTML = `${observation ? `<p><strong>${m('observed')}</strong><br>${escape(Array.isArray(observation) ? observation.join(' ') : observation)}</p>` : ''}${visual ? `<p class="visual-observation"><strong>${m('visual')}</strong><br>${escape(visual)}</p>` : ''}${hypothesis ? `<p class="hypothesis"><strong>${m('hypothesis')}</strong><br>${escape(Array.isArray(hypothesis) ? hypothesis.join(' ') : hypothesis)}</p>` : ''}`;
    renderPredictions();
  }

  function scoreMeaning(type, id) {
    const token = String(type || '').toLowerCase();
    if (id === 'poet_geometry_retrieval' || token.includes('blend')) return m('blend');
    if (id === 'linear_svm' || token.includes('margin')) return m('margin');
    if (id === 'knn' || token.includes('vote')) return m('votes');
    if (id === 'random_forest' || token.includes('forest')) return m('forestProb');
    if (id === 'poet_official_v2' || token.includes('softmax')) return m('softmax');
    return m('scoreGeneric');
  }

  function scoreDisplay(score, id) {
    if (!Number.isFinite(Number(score)) || score === null || score === undefined) return '—';
    const value = Number(score);
    if (id === 'linear_svm') return `${value > 0 ? '+' : ''}${value.toFixed(3)}`;
    if (value > 0 && value < 0.0001) return '<0.01%';
    return `${(value * 100).toFixed(value > 0 && value < 0.001 ? 2 : 1)}%`;
  }

  function renderPredictions() {
    const row = state.uploadName ? null : currentCase();
    const result = state.liveResult;
    const live = !!result;
    const methods = live ? Object.fromEntries(Object.entries(result.methods || {}).map(([key, value]) => [methodId(key), value])) : null;
    const ids = [...order, ...(methods?.poet_geometry_retrieval ? ['poet_geometry_retrieval'] : [])];
    const truth = row?.gloss;
    $('prediction-source').textContent = live ? m('livePredictions') : state.uploadName ? m(state.busy ? 'waiting' : 'noPred') : t('savedPredictions');
    $('prediction-source').classList.toggle('live', live);
    $('predictions').innerHTML = ids.map((id) => {
      const prediction = methods?.[id];
      const topk = live ? prediction?.topk : (!state.uploadName ? row?.top5?.[id] : null);
      const correct = truth && topk?.length ? (live ? topk[0].word === truth : row?.correct?.[id]) : null;
      const kind = typeof correct === 'boolean' ? (correct ? 'correct' : 'wrong') : 'unknown';
      const scoreType = live ? prediction?.score_type : row?.score_types?.[id];
      const list = topk?.length ? `<ol class="topk-list">${topk.slice(0, 5).map((item, i) => {
        const word = item.word ?? item.gloss ?? '';
        return `<li class="${i === 0 ? 'first ' : ''}${word === truth ? 'truth' : ''}"><span class="rank">${i + 1}</span><span class="candidate-word">${escape(word)}</span><span class="candidate-score">${escape(scoreDisplay(item.score, id))}</span></li>`;
      }).join('')}</ol>` : `<p class="empty-text">${state.busy ? m('waiting') : m('noPred')}</p>`;
      return `<article class="prediction-card ${id === 'poet_geometry_retrieval' ? 'improved' : ''}" style="--method-color:${colors[id]}"><header><h4>${names[id]}</h4><span class="correctness ${kind}">${m(kind)}</span></header>${list}<p class="score-type">${scoreMeaning(scoreType, id)}${prediction?.seed ? ` · seed ${escape(prediction.seed)}` : ''}</p></article>`;
    }).join('');
  }

  function renderOverlap() {
    const methodOptions = order.map((id) => [id, names[id]]);
    $('pair-a').innerHTML = options(methodOptions, state.pairA);
    $('pair-b').innerHTML = options(methodOptions, state.pairB);
    if (state.pairA === state.pairB) { $('overlap-grid').innerHTML = `<p class="empty-text">${m('sameMethod')}</p>`; text('overlap-note', ''); return; }
    let row = (data?.pairwise || []).find((item) => item.method_a === state.pairA && item.method_b === state.pairB);
    if (!row) {
      const reverse = (data?.pairwise || []).find((item) => item.method_b === state.pairA && item.method_a === state.pairB);
      if (reverse) row = { ...reverse, only_a_correct: reverse.only_b_correct, only_b_correct: reverse.only_a_correct };
    }
    if (!row) { $('overlap-grid').innerHTML = `<p class="empty-text">${m('noGroups')}</p>`; return; }
    $('overlap-grid').innerHTML = [['bothRight', row.both_correct], ['onlyA', row.only_a_correct], ['onlyB', row.only_b_correct], ['bothWrong', row.both_incorrect]].map(([label, count]) => `<div class="overlap-cell"><strong>${number(count)}</strong><span>${m(label)}${label === 'onlyA' ? ` · ${names[state.pairA]}` : label === 'onlyB' ? ` · ${names[state.pairB]}` : ''}</span></div>`).join('');
    text('overlap-note', m('oracle', { n: number(row.either_correct), total: number(row.count), pct: percent(row.either_correct / row.count) }));
  }

  function renderSigners() {
    $('signer-method').innerHTML = options(order.map((id) => [id, names[id]]), state.signerMethod);
    $('signer-chart').innerHTML = (data?.per_signer || []).map((row) => {
      const value = row.metrics?.[state.signerMethod]?.top1;
      const id = String(row.signer_id);
      return `<div class="signer-row"><div class="signer-label">${escape(id)}<small>n = ${number(row.count)}</small></div><div class="signer-track"><div class="signer-fill" style="--width:${Math.min(100, Math.max(0, Number(value) * 100))}%;background:${colors[state.signerMethod]}"></div></div><span class="signer-score">${percent(value, 1)}</span></div>`;
    }).join('');
  }

  function dimensionLabel(key) {
    const known = { duration: ['视频时长', 'Video duration'], frame_count: ['帧数', 'Frame count'], hand_detection: ['手部检测覆盖率', 'Hand detection coverage'], hand_coverage: ['手部检测覆盖率', 'Hand detection coverage'], both_hand_coverage: ['双手检测覆盖率', 'Both-hand coverage'], any_hand_coverage: ['任一手检测覆盖率', 'Any-hand coverage'], train_support: ['训练支持样本数', 'Training support'], class_frequency: ['训练类别频次', 'Training class frequency'] };
    return known[key]?.[language === 'en' ? 1 : 0] || key.replaceAll('_', ' ');
  }

  function renderGroups() {
    const groups = data?.groups || [];
    const dimensions = [...new Set(groups.map((x) => x.dimension))];
    if (!dimensions.includes(state.groupDimension)) state.groupDimension = dimensions[0];
    $('group-dimension').innerHTML = options(dimensions.map((key) => [key, dimensionLabel(key)]), state.groupDimension);
    const rows = groups.filter((x) => x.dimension === state.groupDimension);
    $('group-table').innerHTML = `<thead><tr><th>${m('group')}</th><th>${m('n')}</th>${order.map((id) => `<th>${names[id]}<br>Top-1</th>`).join('')}</tr></thead><tbody>${rows.map((row) => {
      const best = Math.max(...order.map((id) => Number(row.metrics?.[id]?.top1)));
      return `<tr><td>${escape(localized(row, 'label'))}</td><td>${number(row.count)}</td>${order.map((id) => `<td class="${row.metrics?.[id]?.top1 === best ? 'best' : ''}">${percent(row.metrics?.[id]?.top1)}</td>`).join('')}</tr>`;
    }).join('')}</tbody>`;
    if (!rows.length) $('group-table').innerHTML = `<tbody><tr><td>${m('noGroups')}</td></tr></tbody>`;
  }

  function renderMethods() {
    $('method-cards').innerHTML = order.map((id, i) => {
      const row = getMethod(id);
      const description = localized(row, 'description') || fallbackDescriptions[id][language === 'en' ? 1 : 0];
      const limitation = localized(row, 'limitations');
      return `<article class="method-card" style="--method-color:${colors[id]}"><span class="method-number">0${i + 1}</span><h3>${names[id]}</h3><div class="method-subtitle">${escape(subtitles[id][language === 'en' ? 1 : 0])}</div><p>${escape(description)}</p>${limitation ? `<details><summary>${m('limitations')}</summary><p>${escape(limitation)}</p></details>` : ''}</article>`;
    }).join('');
  }

  function renderAblation() {
    const rows = Array.isArray(data?.factorial?.summary) ? data.factorial.summary : [];
    const labels = { B: 'B · PoET', B_plus_R: 'B + R', G: 'G · PoET + geometry', G_plus_R: 'G + R' };
    $('ablation-table').innerHTML = `<table><thead><tr><th>${m('method')}</th><th>${m('meanSD')}</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${escape(labels[row.id] || row.name || row.id)}</td><td class="${row.id === 'G_plus_R' ? 'best' : ''}">${(row.mean_top1 * 100).toFixed(4)} ± ${(row.std_top1 * 100).toFixed(4)}</td></tr>`).join('')}</tbody></table>`;
  }

  function updateControls() {
    $('run-case').disabled = !state.ready || state.busy || !!state.uploadName || !currentCase();
    $('upload-trigger').disabled = !state.ready || state.busy;
    ['case-filter', 'case-select'].forEach((id) => { $(id).disabled = state.busy; });
    $('previous-case').disabled = state.busy || cases().length < 2;
    $('next-case').disabled = state.busy || cases().length < 2;
    $('connection-pill').classList.toggle('live', state.ready);
    $('connection-pill').innerHTML = `<i></i><span>${state.ready ? (language === 'en' ? en.live : '服务已连接') : t('offline')}</span>`;
    $('language-toggle').textContent = language === 'zh' ? 'EN ↗' : '中文 ↗';
    $('language-toggle').setAttribute('aria-label', language === 'zh' ? 'Switch to English' : '切换为中文');
    $('previous-case').setAttribute('aria-label', language === 'zh' ? '上一个案例' : 'Previous case');
    $('next-case').setAttribute('aria-label', language === 'zh' ? '下一个案例' : 'Next case');
    $('signer-method').setAttribute('aria-label', language === 'zh' ? '选择 signer 分析方法' : 'Method for signer analysis');
    $('group-dimension').setAttribute('aria-label', language === 'zh' ? '选择诊断分组' : 'Diagnostic grouping');
  }

  function renderJob() {
    $('job-status').hidden = !state.jobMessage;
    $('job-status').className = `job-status ${state.jobState === 'failed' ? 'error' : state.busy ? 'running' : ''}`;
    $('job-status').textContent = state.jobMessage ? m(state.jobMessage) : '';
  }

  function renderLiveDetails() {
    const result = state.liveResult;
    $('live-details').hidden = !result;
    if (!result) { $('live-details').innerHTML = ''; return; }
    const coverage = result.coverage || {};
    const duration = result.timings_seconds?.request_total;
    const items = [];
    if (Number.isFinite(duration)) items.push(`${m('elapsed')} ${duration.toFixed(2)} ${m('seconds')}`);
    if (coverage.frame_count !== undefined) items.push(`${m('frames')} ${number(coverage.frame_count)}`);
    if (coverage.left_hand?.fraction !== undefined) items.push(`${m('leftHand')} ${percent(coverage.left_hand.fraction, 1)}`);
    if (coverage.right_hand?.fraction !== undefined) items.push(`${m('rightHand')} ${percent(coverage.right_hand.fraction, 1)}`);
    $('live-details').innerHTML = items.map((item) => `<span>${escape(item)}</span>`).join('') + `<span>${m('coverageNote')}</span>` + (coverage.all_hands_missing ? `<p class="warning">${m('noHands')}</p>` : result.warnings?.length ? `<p class="warning">${m('otherWarnings')}</p>` : '');
  }

  function renderLanguage() {
    document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
    document.querySelectorAll('[data-i18n]').forEach((node) => { node.innerHTML = t(node.dataset.i18n); });
    renderBenchmark(); renderSelectors(); renderCase(false); renderOverlap(); renderSigners(); renderGroups(); renderMethods(); renderAblation(); updateControls(); renderJob(); renderLiveDetails();
    if (state.uploadName) renderUploadInfo();
    else if (state.liveResult) $('case-observations').textContent = m('liveObservation');
  }

  function selectCase(id) {
    if (state.busy) return;
    state.caseId = id;
    state.liveResult = null;
    state.uploadName = null;
    state.jobMessage = null;
    state.jobState = null;
    if (state.uploadUrl) { URL.revokeObjectURL(state.uploadUrl); state.uploadUrl = null; }
    renderSelectors(); renderCase(); updateControls(); renderJob(); renderLiveDetails();
  }

  async function checkConnection() {
    if (location.protocol !== 'http:' && location.protocol !== 'https:') { state.ready = false; updateControls(); return; }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4500);
    try {
      const response = await fetch('/api/status', { cache: 'no-store', signal: controller.signal });
      const status = response.ok ? await response.json() : null;
      state.ready = !!status && status.ready !== false && status.available !== false && status.status !== 'failed';
    } catch (_) { state.ready = false; } finally { clearTimeout(timer); updateControls(); }
  }

  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  async function requestJson(url, options = {}, timeout = 30000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      if (!response.ok) throw new Error('request-failed');
      return await response.json();
    } finally { clearTimeout(timer); }
  }

  async function startJob(payload) {
    if (state.busy || !state.ready) return;
    state.busy = true;
    state.liveResult = null;
    state.jobState = 'queued';
    state.jobMessage = 'queued';
    updateControls(); renderJob(); renderPredictions(); renderLiveDetails();
    try {
      const submitted = await requestJson('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }, 60000);
      if (!submitted.job_id) throw new Error('missing-job-id');
      let job = submitted;
      const started = Date.now();
      while (job.status !== 'completed' && job.status !== 'failed') {
        if (Date.now() - started > 15 * 60 * 1000) throw new Error('timeout');
        await delay(1500);
        job = await requestJson(`/api/jobs/${encodeURIComponent(submitted.job_id)}`, { cache: 'no-store' });
        state.jobState = job.status;
        state.jobMessage = job.status === 'queued' ? 'queued' : 'running';
        renderJob();
      }
      if (job.status !== 'completed' || !job.result?.methods) throw new Error('inference-failed');
      state.liveResult = job.result;
      state.jobState = 'completed';
      state.jobMessage = 'complete';
      if (!state.uploadName) $('case-observations').textContent = m('liveObservation');
    } catch (error) {
      state.jobState = 'failed';
      state.jobMessage = error.message === 'timeout' ? 'timeout' : 'failed';
      checkConnection();
    } finally {
      state.busy = false;
      updateControls(); renderJob(); renderPredictions(); renderLiveDetails();
    }
  }

  function renderUploadInfo() {
    renderSelectors();
    text('truth-caption', m('uploadLabel'));
    text('ground-truth', state.uploadName);
    text('case-source', m('uploaded'));
    $('case-observations').textContent = m('uploadedInfo');
    renderPredictions();
  }

  async function validateVideo(file) {
    if (!file.size || file.size > 25 * 1024 * 1024) throw new Error('sizeError');
    if (!/\.(mp4|mov|webm)$/i.test(file.name)) throw new Error('typeError');
    const url = URL.createObjectURL(file);
    try {
      const duration = await new Promise((resolve, reject) => {
        const video = document.createElement('video');
        const timer = setTimeout(() => reject(new Error('videoError')), 12000);
        video.preload = 'metadata';
        video.onloadedmetadata = () => { clearTimeout(timer); const seconds = video.duration; video.removeAttribute('src'); video.load(); resolve(seconds); };
        video.onerror = () => { clearTimeout(timer); reject(new Error('videoError')); };
        video.src = url;
      });
      if (!Number.isFinite(duration) || duration <= 0) throw new Error('videoError');
      if (duration > 30) throw new Error('durationError');
    } finally { URL.revokeObjectURL(url); }
  }

  async function uploadVideo(file) {
    if (!file || !state.ready || state.busy) return;
    state.busy = true;
    state.jobMessage = 'uploading'; state.jobState = 'queued';
    updateControls(); renderJob();
    try {
      await validateVideo(file);
      state.jobMessage = 'uploading'; state.jobState = 'queued'; renderJob();
      const encoded = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1]);
        reader.onerror = () => reject(new Error('videoError'));
        reader.readAsDataURL(file);
      });
      state.uploadName = file.name;
      state.liveResult = null;
      if (state.uploadUrl) URL.revokeObjectURL(state.uploadUrl);
      state.uploadUrl = URL.createObjectURL(file);
      $('case-video').pause(); $('case-video').src = state.uploadUrl; $('case-video').removeAttribute('poster'); $('case-video').hidden = false; $('video-placeholder').hidden = true; $('case-video').load();
      $('frame-strip').innerHTML = '';
      renderUploadInfo();
      state.busy = false;
      await startJob({ filename: file.name, video_base64: encoded });
    } catch (error) {
      state.jobState = 'failed'; state.jobMessage = messages[error.message] ? error.message : 'failed'; renderJob();
    } finally { state.busy = false; updateControls(); renderJob(); $('upload-file').value = ''; }
  }

  $('language-toggle').addEventListener('click', () => { language = language === 'zh' ? 'en' : 'zh'; try { localStorage.setItem('asl-lab-language', language); } catch (_) { /* Optional preference persistence. */ } renderLanguage(); });
  document.querySelectorAll('[data-metric]').forEach((button) => button.addEventListener('click', () => { state.metric = button.dataset.metric; renderBenchmark(); }));
  $('case-filter').addEventListener('change', (event) => { state.filter = event.target.value; selectCase(cases()[0]?.id); });
  $('case-select').addEventListener('change', (event) => selectCase(event.target.value));
  const step = (direction) => { const rows = cases(); if (!rows.length) return; const index = rows.findIndex((row) => row.id === state.caseId); selectCase(rows[(index + direction + rows.length) % rows.length].id); };
  $('previous-case').addEventListener('click', () => step(-1)); $('next-case').addEventListener('click', () => step(1));
  $('pair-a').addEventListener('change', (event) => { state.pairA = event.target.value; renderOverlap(); });
  $('pair-b').addEventListener('change', (event) => { state.pairB = event.target.value; renderOverlap(); });
  $('signer-method').addEventListener('change', (event) => { state.signerMethod = event.target.value; renderSigners(); });
  $('group-dimension').addEventListener('change', (event) => { state.groupDimension = event.target.value; renderGroups(); });
  $('run-case').addEventListener('click', () => { if (currentCase()) startJob({ case_id: currentCase().id }); });
  $('upload-trigger').addEventListener('click', () => $('upload-file').click());
  $('upload-file').addEventListener('change', (event) => uploadVideo(event.target.files[0]));
  $('frame-strip').addEventListener('click', (event) => { const button = event.target.closest('[data-frame]'); const video = $('case-video'); if (button && Number.isFinite(video.duration)) { video.currentTime = video.duration * Number(button.dataset.frame) / 100; video.pause(); } });
  $('case-video').addEventListener('error', () => { $('case-video').hidden = true; $('video-placeholder').hidden = false; });
  if (!data) {
    state.jobState = 'failed'; state.jobMessage = 'noData'; renderJob(); updateControls();
  } else {
    renderLanguage(); renderCase(); checkConnection();
    if (location.protocol === 'http:' || location.protocol === 'https:') setInterval(() => { if (!state.busy) checkConnection(); }, 30000);
  }
})();
