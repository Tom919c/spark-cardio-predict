(function () {
  'use strict';

  const LOCAL_ECHARTS = '/static/echarts/echarts.min.js';

  function loadScript(url) {
    return new Promise((resolve, reject) => {
      const el = document.createElement('script');
      el.src = url;
      el.async = true;
      el.onload = () => resolve(url);
      el.onerror = () => reject(new Error(url));
      document.head.appendChild(el);
    });
  }

  async function ensureECharts() {
    if (window.echarts) return window.echarts;
    try {
      await loadScript(LOCAL_ECHARTS);
    } catch (localErr) {
      console.warn('本地 ECharts 资源不可用，当前页面将显示空态。', localErr);
    }
    return window.echarts;
  }

  function safeJson(text) {
    try { return JSON.parse(text); } catch (_) { return null; }
  }

  function fmtPercent(value) {
    if (value === null || value === undefined || value === '') return '--';
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(1) + '%' : '--';
  }

  function setStatus(el, kind, text) {
    if (!el) return;
    el.className = kind ? ('phase2-' + kind) : '';
    el.textContent = text;
  }

  function renderEmpty(dom, text, kind = 'empty') {
    if (!dom) return;
    dom.innerHTML = '<div class="phase2-' + kind + '">' + text + '</div>';
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function renderUploadResult(result) {
    const target = document.getElementById('phase2UploadResult');
    if (!target) return;
    const data = result || {};
    const metricSource = data.risk_metric_source || 'unavailable';
    const metricLabel = metricSource === 'labels'
      ? '事件标签率'
      : (metricSource === 'model_probability' ? '模型高风险率' : '风险率');
    const regions = Array.isArray(data.region_analysis)
      ? data.region_analysis
      : (Array.isArray(data.districts) ? data.districts : []);
    const rows = regions.map(function (item) {
      return '<tr><td>' + escapeHtml(item.region || item.district || item.community || '未知') +
        '</td><td>' + (item.residents == null ? '已匿名' : escapeHtml(item.residents)) +
        '</td><td>' + fmtPercent(item.heart_rate) +
        '</td><td>' + fmtPercent(item.stroke_rate) + '</td></tr>';
    }).join('');
    target.innerHTML =
      '<div class="phase2-muted">覆盖口径：' + escapeHtml(data.coverage_name || '--') +
      ' · ' + escapeHtml(data.coverage_level || '--') + '</div>' +
      '<div class="phase2-result-metrics">' +
        '<div class="phase2-result-metric"><span class="phase2-muted">居民数</span><strong>' + escapeHtml(data.total_residents ?? '--') + '</strong></div>' +
        '<div class="phase2-result-metric"><span class="phase2-muted">心脏' + metricLabel + '</span><strong>' + fmtPercent(data.heart_risk_rate) + '</strong></div>' +
        '<div class="phase2-result-metric"><span class="phase2-muted">卒中' + metricLabel + '</span><strong>' + fmtPercent(data.stroke_risk_rate) + '</strong></div>' +
        '<div class="phase2-result-metric"><span class="phase2-muted">双事件' + metricLabel + '</span><strong>' + fmtPercent(data.comorbidity_rate) + '</strong></div>' +
      '</div>' +
      (rows ? '<div class="phase2-result-table"><table class="phase2-table"><thead><tr><th>区域</th><th>人数</th><th>心脏</th><th>卒中</th></tr></thead><tbody>' + rows + '</tbody></table></div>' : '<div class="phase2-empty" style="margin-top:10px;">暂无可展示的区域明细。</div>');
  }

  async function fetchJson(url, options = {}) {
    const resp = await fetch(url, Object.assign({ headers: { Accept: 'application/json' } }, options));
    const data = await resp.json().catch(() => null);
    if (!resp.ok || !data || data.success === false) {
      throw new Error((data && data.message) || ('HTTP ' + resp.status));
    }
    return data.data ?? data;
  }

  function renderSnapshotTrend(snapshots) {
    const box = document.getElementById('snapshotTrendChart');
    if (!box || !window.echarts) return;
    const rows = Array.isArray(snapshots) ? snapshots : [];
    const labels = rows.map(item => item.data_period);
    const heart = rows.map(item => item.heart_risk_rate == null ? null : Number(item.heart_risk_rate));
    const stroke = rows.map(item => item.stroke_risk_rate == null ? null : Number(item.stroke_risk_rate));
    const chart = echarts.init(box);
    if (!rows.length) {
      chart.setOption({ title: { text: '暂无快照趋势', left: 'center', top: 'center', textStyle: { color: '#94a3b8', fontSize: 13 } } });
      return;
    }
    chart.setOption({
      tooltip: { trigger: 'axis' },
      grid: { left: 8, right: 10, top: 20, bottom: 20, containLabel: true },
      xAxis: { type: 'category', data: labels, axisLabel: { color: '#cbd5e1' } },
      yAxis: { type: 'value', axisLabel: { color: '#cbd5e1', formatter: '{value}%' } },
      series: [
        { name: '心脏高风险率', type: 'line', smooth: true, symbolSize: 8, data: heart, lineStyle: { color: '#60a5fa', width: 3 }, itemStyle: { color: '#93c5fd' } },
        { name: '卒中高风险率', type: 'line', smooth: true, symbolSize: 8, data: stroke, lineStyle: { color: '#fb923c', width: 3 }, itemStyle: { color: '#fdba74' } }
      ]
    });
  }

  async function initDashboardEnhancement() {
    const panel = document.getElementById('phase2UploadPanel');
    const progress = document.getElementById('phase2UploadProgress');
    const status = document.getElementById('phase2UploadStatus');
    const fileInput = document.getElementById('phase2UploadFile');
    const taskIdEl = document.getElementById('phase2UploadTaskId');
    const mapMode = document.getElementById('phase2MapMode');
    const regionTable = document.getElementById('phase2RegionTable');
    const trendReady = document.getElementById('snapshotTrendChart');

    if (!panel && !trendReady && !mapMode && !regionTable) return;
    await ensureECharts();

    if (trendReady && window.echarts) {
      fetchJson('/api/trends/snapshots').then(function(result) {
        const snapshots = result.snapshots || [];
        if (snapshots.length < 2) {
          renderSnapshotTrend(snapshots);
          return null;
        }
        return fetchJson('/api/trends/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ snapshots: snapshots })
        }).then(function(comparison) {
          renderSnapshotTrend((comparison.series || []).map(function(item) {
            return Object.assign({}, item, {
              risk_metric_source: item.risk_metric_source || 'unavailable'
            });
          }));
        });
      }).catch(function() {
        renderSnapshotTrend([]);
      });
    }

    if (mapMode || regionTable) {
      try {
        const dashboard = await fetchJson('/api/analysis/dashboard');
        const coverage = dashboard.coverage_level || dashboard.map_coverage_level || 'map';
        const mapAvailable = dashboard.map_available !== false;
        const regions = Array.isArray(dashboard.region_analysis) ? dashboard.region_analysis : (Array.isArray(dashboard.regions) ? dashboard.regions : (Array.isArray(dashboard.districts) ? dashboard.districts : []));
        if (mapMode) {
          mapMode.textContent = dashboard.has_data === false
            ? '等待上传居民数据'
            : (mapAvailable && coverage !== 'table' ? '地图模式' : '表格模式');
        }
        if (regionTable) {
          if (regions.length) {
            regionTable.innerHTML = '<table class="phase2-table"><thead><tr><th>区域</th><th>人数</th><th>心脏风险率</th><th>卒中风险率</th></tr></thead><tbody>' + regions.map(r => '<tr><td>' + escapeHtml(r.name || r.region || r.district || r.community || '--') + '</td><td>' + (r.residents == null ? '已匿名' : escapeHtml(r.residents)) + '</td><td>' + fmtPercent(r.heart_rate) + '</td><td>' + fmtPercent(r.stroke_rate) + '</td></tr>').join('') + '</tbody></table>';
          } else {
            renderEmpty(regionTable, '当前没有可展示的区域明细，后端返回空列表时这里会自动降级为表格空态。');
          }
        }
      } catch (err) {
        if (mapMode) setStatus(mapMode, 'failure', '区域分析读取失败：' + err.message);
        if (regionTable) renderEmpty(regionTable, '区域分析读取失败，可能是后端未启动或接口暂不可用。', 'failure');
      }
    }

    if (panel && fileInput) {
      const endpoint = window.PHASE2_UPLOAD_ENDPOINT || '/api/upload/tasks';
      const pollEndpoint = window.PHASE2_TASK_STATUS_ENDPOINT || '/api/tasks/status';
      const startBtn = document.getElementById('phase2UploadStart');
      const chunkEl = document.getElementById('phase2UploadChunk');
      const chunkProgress = document.getElementById('phase2UploadChunkProgress');
      let selectedFile = null;

      fileInput.addEventListener('change', () => {
        selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
        if (status) status.textContent = selectedFile ? ('已选择：' + selectedFile.name) : '请选择文件后开始上传';
      });

      async function pollTask(taskId) {
        if (!taskId) return;
        const deadline = Date.now() + 180000;
        while (Date.now() < deadline) {
          try {
            const task = await fetchJson(pollEndpoint + '/' + encodeURIComponent(taskId));
            const state = String(task.state || task.status || '').toLowerCase();
            if (taskIdEl) taskIdEl.textContent = task.task_id || taskId;
            if (status) status.textContent = task.message || ('任务状态：' + (state || 'running'));
            if (chunkProgress) chunkProgress.style.width = (Number(task.progress ?? task.percent ?? 0)) + '%';
            const completed = (state === 'success' || state === 'completed' || state === 'done') &&
              (task.stage === 'completed' || task.result_path);
            if (completed) {
              if (task.dataset_id) {
                try {
                  const result = await fetchJson('/api/datasets/' + encodeURIComponent(task.dataset_id) + '/result');
                  renderUploadResult(result);
                  sessionStorage.setItem('cardiospark_latest_result', JSON.stringify(result));
                  if (typeof window.refreshInstitutionDashboard === 'function') {
                    window.refreshInstitutionDashboard();
                  }
                } catch (resultErr) {
                  renderEmpty(document.getElementById('phase2UploadResult'), '任务已完成，但分析结果读取失败：' + escapeHtml(resultErr.message), 'failure');
                }
              }
              return task;
            }
            if (state === 'failed' || state === 'error') throw new Error(task.message || '任务失败');
          } catch (err) {
            setStatus(status, 'failure', '任务轮询失败：' + err.message);
            return;
          }
          await new Promise(r => setTimeout(r, 3000));
        }
        setStatus(status, 'offline', '任务轮询超时，页面会保留当前进度结果。');
      }

      if (startBtn) {
        startBtn.addEventListener('click', async () => {
          if (!selectedFile) {
            setStatus(status, 'empty', '请先选择待上传文件。');
            return;
          }
          if (progress) progress.style.width = '0%';
          if (chunkProgress) chunkProgress.style.width = '0%';
          setStatus(status, '', '开始分片上传……');
          try {
            const chunkSize = Number(chunkEl && chunkEl.value) || (8 * 1024 * 1024);
            const totalChunks = Math.ceil(selectedFile.size / chunkSize);
            const periodInput = document.getElementById('phase2DataPeriod');
            const dataPeriod = String(periodInput?.value || '').trim();
            if (!/^\d{4}-(?:0[1-9]|1[0-2]|Q[1-4])$/.test(dataPeriod)) {
              throw new Error('请填写正确的数据所属时期，例如 2026-07 或 2026-Q3。');
            }
            const initResp = await fetch(endpoint, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
              body: JSON.stringify({
                filename: selectedFile.name,
                file_size: selectedFile.size,
                total_chunks: totalChunks,
                data_period: dataPeriod
              })
            });
            const resp = initResp;
            const json = await resp.json().catch(() => null);
            if (!resp.ok || !json || json.success === false) throw new Error((json && json.message) || ('HTTP ' + resp.status));
            const taskId = (json.data && (json.data.task_id || json.data.id)) || json.task_id || json.id;
            if (taskIdEl) taskIdEl.textContent = taskId || '--';
            for (let index = 0; index < totalChunks; index += 1) {
              const start = index * chunkSize;
              const chunk = selectedFile.slice(start, Math.min(start + chunkSize, selectedFile.size));
              const form = new FormData();
              form.append('chunk', chunk, selectedFile.name + '.part');
              form.append('chunk_index', String(index));
              const chunkResponse = await fetch('/api/upload/tasks/' + encodeURIComponent(taskId) + '/chunks', { method: 'POST', body: form });
              const chunkJson = await chunkResponse.json().catch(() => null);
              if (!chunkResponse.ok || !chunkJson || chunkJson.success === false) throw new Error((chunkJson && chunkJson.message) || ('HTTP ' + chunkResponse.status));
              const percent = ((index + 1) / totalChunks * 100).toFixed(1);
              if (progress) progress.style.width = percent + '%';
              if (chunkProgress) chunkProgress.style.width = percent + '%';
              setStatus(status, 'banner', '正在上传第 ' + (index + 1) + '/' + totalChunks + ' 个分片……');
            }
            const finalizeResponse = await fetch('/api/upload/tasks/' + encodeURIComponent(taskId) + '/finalize', { method: 'POST', headers: { Accept: 'application/json' } });
            const finalizeJson = await finalizeResponse.json().catch(() => null);
            if (!finalizeResponse.ok || !finalizeJson || finalizeJson.success === false) throw new Error((finalizeJson && finalizeJson.message) || ('HTTP ' + finalizeResponse.status));
            setStatus(status, 'banner', '上传完成，后台正在分析……');
            await pollTask(taskId);
          } catch (err) {
            setStatus(status, 'failure', '上传失败：' + err.message);
          }
        });
      }
    }
  }

  async function loadPlatformCapabilities() {
    const database = document.getElementById('phase2DatabaseEngine');
    const storage = document.getElementById('phase2StorageEngine');
    const compute = document.getElementById('phase2ComputeEngine');
    if (!database && !storage && !compute) return;
    try {
      const payload = await fetchJson('/api/capabilities');
      const data = payload.data || {};
      if (database) database.textContent = String(data.database?.type || 'sqlite').toUpperCase();
      if (storage) storage.textContent = data.storage_engine === 'hdfs' ? 'HDFS / WebHDFS' : '本地文件系统';
      if (compute) compute.textContent = data.compute_engine === 'apache_spark' ? 'Apache Spark' : 'Pandas 本地引擎';
    } catch (_) {
      if (database) database.textContent = '状态未知';
      if (storage) storage.textContent = '状态未知';
      if (compute) compute.textContent = '状态未知';
    }
  }

  async function initResultEnhancement() {
    const target = document.getElementById('phase2WhatIf');
    if (!target) return;
    const raw = sessionStorage.getItem('cardiospark_latest_result');
    const data = raw ? safeJson(raw) : null;
    if (!data) {
      renderEmpty(target, '尚无最近一次评估结果，What-if 结果会在这里展示。');
      return;
    }
    const whatIf = data.what_if_result || data.what_if || null;
    if (!whatIf) {
      renderEmpty(target, '当前结果中没有后端返回的 What-if 试算数据。若后端后续返回该字段，这里会自动展示。');
      return;
    }
    const items = Array.isArray(whatIf) ? whatIf : [whatIf];
    target.innerHTML = '<div class="phase2-grid phase2-grid--2">' + items.map(item => '<div class="phase2-card"><div class="phase2-title">' + (item.title || item.label || 'What-if 方案') + '</div><div class="phase2-muted">' + (item.summary || item.desc || '--') + '</div><div class="phase2-pill" style="margin-top:10px;">结果：' + (item.result || item.value || '--') + '</div></div>').join('') + '</div>';
  }

  async function initShapEnhancement() {
    const empty = document.getElementById('phase2ShapState');
    if (!empty) return;
    const raw = sessionStorage.getItem('cardiospark_latest_result');
    const data = raw ? safeJson(raw) : null;
    if (!data) {
      renderEmpty(empty, '当前没有最近一次评估的原因说明结果。请先完成一次预测。');
      return;
    }
    const shap = data.combined_shap_summary || {};
    if (!shap.available) {
      renderEmpty(empty, '后端暂未返回 SHAP 结果，页面将显示空态而不会伪造分析结论。');
    }
  }

  async function boot() {
    await ensureECharts();
    await loadPlatformCapabilities();
    initDashboardEnhancement();
    initResultEnhancement();
    initShapEnhancement();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
