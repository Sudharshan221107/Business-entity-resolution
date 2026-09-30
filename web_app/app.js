document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  // Theme Manager
  const themeSelect = document.getElementById("theme-select");
  const savedTheme = localStorage.getItem("er_theme") || "cyber-emerald";
  document.documentElement.setAttribute("data-theme", savedTheme);
  document.body.setAttribute("data-theme", savedTheme);
  if (themeSelect) {
    themeSelect.value = savedTheme;
    themeSelect.addEventListener("change", (e) => {
      const selected = e.target.value;
      document.documentElement.setAttribute("data-theme", selected);
      document.body.setAttribute("data-theme", selected);
      localStorage.setItem("er_theme", selected);
    });
  }

  // State
  let allRecords = [];
  let isAutoscroll = true;
  let pipelinePollInterval = null;

  // DOM Elements
  const datasetReadyPill = document.getElementById("dataset-ready-pill");
  const engineStatusPill = document.getElementById("engine-status-pill");
  const btnRunPipeline = document.getElementById("btn-run-pipeline");
  const runBtnText = document.getElementById("run-btn-text");
  const pipelineProgressBar = document.getElementById("pipeline-progress-bar");
  const pipelinePctLabel = document.getElementById("pipeline-pct-label");
  const pipelineStageLabel = document.getElementById("pipeline-stage-label");
  const terminalLogs = document.getElementById("terminal-logs");
  const btnClearLogs = document.getElementById("btn-clear-logs");
  const btnScrollBottom = document.getElementById("btn-scroll-bottom");

  // Search & Filter Elements
  const searchInput = document.getElementById("search-input");
  const countryFilter = document.getElementById("country-filter");
  const typeFilter = document.getElementById("type-filter");
  const recordsList = document.getElementById("records-list");
  const recordCount = document.getElementById("record-count");
  const viewCardsBtn = document.getElementById("view-cards-btn");
  const viewTableBtn = document.getElementById("view-table-btn");
  const tableViewContainer = document.getElementById("table-view-container");
  const resultsTableBody = document.getElementById("results-table-body");
  const btnDownloadTableCsv = document.getElementById("btn-download-table-csv");

  // Modal Elements
  const previewModal = document.getElementById("preview-modal");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalFileTitle = document.getElementById("modal-file-title");
  const modalMetaLabel = document.getElementById("modal-meta-label");
  const previewThead = document.getElementById("preview-thead");
  const previewTbody = document.getElementById("preview-tbody");

  // --- 1. DATASET STATUS & REPOSITORY ---
  function fetchDatasetStatus() {
    fetch("/api/dataset_status")
      .then(res => res.json())
      .then(data => {
        let readyCount = 0;
        const total = 7;
        for (const [key, info] of Object.entries(data.files)) {
          const card = document.querySelector(`.dataset-card[data-key="${key}"]`);
          if (card) {
            const badge = card.querySelector(".file-badge");
            const rowsEl = card.querySelector(".val-rows");
            const sizeEl = card.querySelector(".val-size");

            if (info.exists && info.rows > 0) {
              readyCount++;
              card.classList.remove("is-missing");
              badge.className = "file-badge badge-ready";
              badge.textContent = "Ready";
              rowsEl.textContent = info.rows.toLocaleString();
              sizeEl.textContent = info.size_str;
            } else {
              card.classList.add("is-missing");
              badge.className = "file-badge badge-missing";
              badge.textContent = "Missing";
              rowsEl.textContent = "0";
              sizeEl.textContent = "0 B";
            }
          }
        }

        datasetReadyPill.textContent = `${readyCount}/${total} Ready`;
        if (readyCount === total) {
          datasetReadyPill.style.background = "var(--color-emerald-bg)";
          datasetReadyPill.style.color = "var(--color-emerald)";
        } else {
          datasetReadyPill.style.background = "var(--color-rose-bg)";
          datasetReadyPill.style.color = "var(--color-rose)";
        }
      })
      .catch(err => console.error("Error fetching dataset status:", err));
  }

  // Restore Benchmark Data Button
  const btnRestoreBenchmark = document.getElementById("btn-restore-benchmark");
  if (btnRestoreBenchmark) {
    btnRestoreBenchmark.addEventListener("click", () => {
      btnRestoreBenchmark.disabled = true;
      btnRestoreBenchmark.textContent = "Restoring...";
      fetch("/api/restore_benchmark", { method: "POST" })
        .then(res => res.json())
        .then(() => {
          btnRestoreBenchmark.disabled = false;
          btnRestoreBenchmark.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
            Reset to Benchmark Data
          `;
          fetchDatasetStatus();
          fetchStats();
          loadRecords();
          alert("Benchmark datasets (US, India, France) successfully restored!");
        })
        .catch(err => {
          btnRestoreBenchmark.disabled = false;
          alert("Failed to restore benchmark: " + err.message);
        });
    });
  }

  // Refresh Datasets Button
  const btnRefreshDatasets = document.getElementById("btn-refresh-datasets");
  if (btnRefreshDatasets) {
    btnRefreshDatasets.addEventListener("click", () => {
      fetchDatasetStatus();
    });
  }

  // ZIP Drag and Drop
  const zipDropzone = document.getElementById("zip-dropzone");
  const zipFileInput = document.getElementById("zip-file-input");
  const btnBrowseZip = document.getElementById("btn-browse-zip");

  if (zipDropzone && zipFileInput) {
    btnBrowseZip.addEventListener("click", (e) => {
      e.stopPropagation();
      zipFileInput.click();
    });

    zipDropzone.addEventListener("click", () => {
      zipFileInput.click();
    });

    zipDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zipDropzone.classList.add("drag-over");
    });

    zipDropzone.addEventListener("dragleave", () => {
      zipDropzone.classList.remove("drag-over");
    });

    zipDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      zipDropzone.classList.remove("drag-over");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleZipUpload(e.dataTransfer.files[0]);
      }
    });

    zipFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleZipUpload(e.target.files[0]);
      }
    });
  }

  function handleZipUpload(file) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
      alert("Please upload a valid .zip dataset archive.");
      return;
    }

    const dropzoneContent = zipDropzone.querySelector(".dropzone-content");
    const origHtml = dropzoneContent.innerHTML;
    dropzoneContent.innerHTML = `
      <div class="upload-icon-circle"><div class="badge-loading" style="padding: 10px;">Uploading...</div></div>
      <h3>Extracting and Mapping Archive...</h3>
      <p>${escapeHtml(file.name)} (${(file.size / 1024).toFixed(1)} KB)</p>
    `;

    fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file
    })
      .then(res => res.json())
      .then(data => {
        dropzoneContent.innerHTML = origHtml;
        // Re-attach listener
        document.getElementById("btn-browse-zip").addEventListener("click", (e) => {
          e.stopPropagation();
          zipFileInput.click();
        });
        if (data.status === "success") {
          fetchDatasetStatus();
          alert(data.message || "ZIP archive processed successfully!");
        } else {
          alert("Error: " + (data.error || "Upload failed"));
        }
      })
      .catch(err => {
        dropzoneContent.innerHTML = origHtml;
        alert("Upload error: " + err.message);
      });
  }

  // Individual TSV Upload Buttons
  document.querySelectorAll(".btn-file-upload").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const container = e.target.closest(".card-footer-actions");
      const hiddenInput = container.querySelector(".hidden-file-input");
      if (hiddenInput) hiddenInput.click();
    });
  });

  document.querySelectorAll(".hidden-file-input").forEach(input => {
    input.addEventListener("change", (e) => {
      const targetKey = e.target.getAttribute("data-target");
      const file = e.target.files[0];
      if (!file) return;

      const card = document.querySelector(`.dataset-card[data-key="${targetKey}"]`);
      const badge = card ? card.querySelector(".file-badge") : null;
      if (badge) {
        badge.className = "file-badge badge-loading";
        badge.textContent = "Uploading...";
      }

      fetch(`/api/upload?target=${encodeURIComponent(targetKey)}&filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: file
      })
        .then(res => res.json())
        .then(data => {
          if (data.status === "success") {
            fetchDatasetStatus();
          } else {
            alert("Upload failed: " + (data.error || "Unknown error"));
            fetchDatasetStatus();
          }
        })
        .catch(err => {
          alert("Upload failed: " + err.message);
          fetchDatasetStatus();
        });
    });
  });

  // TSV Table Preview
  document.querySelectorAll(".btn-file-preview").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const targetKey = e.target.getAttribute("data-target");
      openPreviewModal(targetKey);
    });
  });

  function openPreviewModal(fileKey) {
    modalFileTitle.textContent = `${fileKey}.tsv Preview`;
    modalMetaLabel.textContent = "Loading preview rows...";
    previewThead.innerHTML = "";
    previewTbody.innerHTML = `<tr><td style="padding: 24px; text-align: center; color: var(--text-subtle);">Fetching dataset preview...</td></tr>`;
    previewModal.classList.remove("hidden");

    fetch(`/api/dataset_preview?file=${encodeURIComponent(fileKey)}`)
      .then(res => res.json())
      .then(data => {
        if (!data.exists) {
          previewTbody.innerHTML = `<tr><td style="padding: 24px; text-align: center; color: var(--color-rose);">File not found or empty.</td></tr>`;
          return;
        }

        modalFileTitle.textContent = `${data.filename} Preview`;
        modalMetaLabel.textContent = `Showing first ${data.rows.length} of ${data.total_rows.toLocaleString()} rows (${data.size_str})`;

        // Build header
        let thHtml = "<tr>";
        data.headers.forEach(h => {
          thHtml += `<th>${escapeHtml(h)}</th>`;
        });
        thHtml += "</tr>";
        previewThead.innerHTML = thHtml;

        // Build rows
        let tbHtml = "";
        data.rows.forEach(r => {
          tbHtml += "<tr>";
          r.forEach(cell => {
            tbHtml += `<td>${escapeHtml(cell)}</td>`;
          });
          tbHtml += "</tr>";
        });
        previewTbody.innerHTML = tbHtml;
      })
      .catch(err => {
        previewTbody.innerHTML = `<tr><td style="padding: 24px; text-align: center; color: var(--color-rose);">Error: ${err.message}</td></tr>`;
      });
  }

  modalCloseBtn.addEventListener("click", () => {
    previewModal.classList.add("hidden");
  });

  previewModal.addEventListener("click", (e) => {
    if (e.target === previewModal) {
      previewModal.classList.add("hidden");
    }
  });

  // --- 2. PIPELINE RUNNER & LIVE LOG STREAMING ---
  function updateProgressUI(pct, stepName, isRunning) {
    pipelineProgressBar.style.width = `${pct}%`;
    pipelinePctLabel.textContent = `${pct}%`;
    pipelineStageLabel.textContent = stepName;

    if (isRunning) {
      btnRunPipeline.disabled = true;
      runBtnText.textContent = "Pipeline Running...";
      engineStatusPill.textContent = "Executing";
      engineStatusPill.className = "tab-pill pill-pulse";
      engineStatusPill.style.background = "var(--color-amber-bg)";
      engineStatusPill.style.color = "var(--color-amber)";
    } else {
      btnRunPipeline.disabled = false;
      runBtnText.textContent = pct === 100 ? "Re-Run Entity Resolution Pipeline" : "Execute Complete Pipeline";
      engineStatusPill.textContent = pct === 100 ? "Finished" : "Ready";
      engineStatusPill.className = "tab-pill";
      if (pct === 100) {
        engineStatusPill.style.background = "var(--color-emerald-bg)";
        engineStatusPill.style.color = "var(--color-emerald)";
      }
    }
  }

  btnRunPipeline.addEventListener("click", () => {
    btnRunPipeline.disabled = true;
    runBtnText.textContent = "Initiating Pipeline...";

    fetch("/api/run", { method: "POST" })
      .then(res => res.json())
      .then(data => {
        if (data.status === "started") {
          startLogPolling();
        } else {
          alert("Could not start pipeline: " + (data.error || "Unknown"));
          btnRunPipeline.disabled = false;
        }
      })
      .catch(err => {
        alert("Execution error: " + err.message);
        btnRunPipeline.disabled = false;
      });
  });

  function startLogPolling() {
    if (pipelinePollInterval) clearInterval(pipelinePollInterval);

    pipelinePollInterval = setInterval(() => {
      fetch("/api/logs")
        .then(res => res.json())
        .then(data => {
          updateProgressUI(data.progress, data.current_step, data.is_running);

          // Render logs
          renderLogs(data.logs);

          if (!data.is_running && data.progress >= 95) {
            clearInterval(pipelinePollInterval);
            pipelinePollInterval = null;
            fetchStats();
            loadRecords();
          }
        })
        .catch(err => console.error("Error polling logs:", err));
    }, 750);
  }

  function renderLogs(logs) {
    if (!logs || logs.length === 0) return;

    terminalLogs.innerHTML = logs.map(line => {
      let cls = "log-line";
      if (line.includes("PASS") || line.includes("Complete!") || line.includes("satisfied")) {
        cls += " text-pass";
      } else if (line.includes("ERROR") || line.includes("FAIL")) {
        cls += " text-error";
      } else if (line.includes("Step")) {
        cls += " text-warn";
      }
      return `<div class="${cls}">${escapeHtml(line)}</div>`;
    }).join("");

    if (isAutoscroll) {
      terminalLogs.scrollTop = terminalLogs.scrollHeight;
    }
  }

  btnClearLogs.addEventListener("click", () => {
    terminalLogs.innerHTML = `<div class="log-line text-muted">[Console cleared]</div>`;
  });

  btnScrollBottom.addEventListener("click", () => {
    isAutoscroll = !isAutoscroll;
    btnScrollBottom.textContent = isAutoscroll ? "Autoscroll: ON" : "Autoscroll: OFF";
    if (isAutoscroll) terminalLogs.scrollTop = terminalLogs.scrollHeight;
  });

  // --- 3. EVALUATION METRICS ---
  function fetchStats() {
    fetch("/api/stats")
      .then(res => res.json())
      .then(data => {
        document.getElementById("stat-f05").textContent = data.macro_f05.toFixed(4);
        document.getElementById("stat-prec").textContent = `${data.precision.toFixed(2)}%`;
        document.getElementById("stat-rec").textContent = `${data.recall.toFixed(2)}%`;
        document.getElementById("stat-block-recall").textContent = `${data.blocking_recall.toFixed(1)}%`;
        document.getElementById("stat-reduction").textContent = `${data.reduction_ratio.toFixed(2)}%`;
        document.getElementById("stat-threshold").textContent = (data.optimal_threshold || 0.375).toFixed(3);

        const vBadge = document.getElementById("validator-badge");
        if (vBadge) {
          vBadge.textContent = `VALIDATION: ${data.validator_status || "PASS"}`;
          vBadge.className = data.validator_status === "PASS" ? "badge status-pass" : "badge status-missing";
        }
      })
      .catch(err => console.error("Error fetching stats:", err));
  }

  // --- 4. RESOLUTION RESULTS EXPLORER ---
  function loadRecords() {
    recordsList.innerHTML = `<div class="loading">Loading resolved entity records...</div>`;
    const q = encodeURIComponent(searchInput.value.trim());
    const country = encodeURIComponent(countryFilter.value);

    fetch(`/api/test_records?q=${q}&country=${country}`)
      .then(res => res.json())
      .then(data => {
        allRecords = data;
        renderFilteredRecords();
      })
      .catch(err => {
        recordsList.innerHTML = `<div class="loading text-rose">Failed to load records: ${err.message}</div>`;
      });
  }

  function renderFilteredRecords() {
    const typeVal = typeFilter.value;
    let filtered = allRecords;

    if (typeVal === "matched") {
      filtered = filtered.filter(r => !r.is_singleton);
    } else if (typeVal === "singletons") {
      filtered = filtered.filter(r => r.is_singleton);
    }

    recordCount.textContent = filtered.length;

    // Card View Rendering
    if (filtered.length === 0) {
      recordsList.innerHTML = `<div class="loading">No matching entities found for the selected criteria.</div>`;
      resultsTableBody.innerHTML = `<tr><td colspan="7" style="padding: 24px; text-align: center; color: var(--text-subtle);">No matching entities found.</td></tr>`;
      return;
    }

    recordsList.innerHTML = filtered.map(rec => {
      const matchBadge = rec.is_singleton
        ? `<span class="match-summary-badge badge-singleton">Singleton (0 Matches)</span>`
        : `<span class="match-summary-badge badge-matched">${rec.matches.length} Match${rec.matches.length > 1 ? "es" : ""} Resolved</span>`;

      const candidateCards = (rec.candidates && rec.candidates.length > 0)
        ? rec.candidates.map(c => `
            <div class="cand-item ${c.is_matched ? 'is-matched' : ''}">
              <div class="cand-top">
                <span class="cand-id">${c.entity_id}</span>
                <span class="cand-status-pill ${c.is_matched ? 'pill-matched' : 'pill-candidate'}">
                  ${c.is_matched ? 'RESOLVED MATCH' : 'BLOCKING CANDIDATE'}
                </span>
              </div>
              <div class="cand-name">${escapeHtml(c.business_name)}</div>
              <div class="cand-addr">${escapeHtml(c.business_address)}</div>
            </div>
          `).join("")
        : `<div class="helper-text" style="padding: 8px;">No candidates generated by blocker.</div>`;

      return `
        <article class="record-card">
          <div class="record-top">
            <div class="entity-identity">
              <div class="id-badge-row">
                <span class="entity-id">${rec.entity_id}</span>
                <span class="country-tag">${escapeHtml(rec.country)}</span>
              </div>
              <h3 class="entity-name">${escapeHtml(rec.business_name)}</h3>
              <p class="entity-address">${escapeHtml(rec.business_address)}</p>
            </div>
            ${matchBadge}
          </div>

          <div class="candidate-section">
            <span class="candidate-header-label">Candidate Set (${rec.candidates.length} evaluated &bull; ${rec.matches.length} predicted match)</span>
            <div class="candidate-grid">
              ${candidateCards}
            </div>
          </div>
        </article>
      `;
    }).join("");

    // Table View Rendering
    resultsTableBody.innerHTML = filtered.map(rec => {
      const statusPill = rec.is_singleton
        ? `<span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-subtle);">Singleton</span>`
        : `<span class="badge status-pass">${rec.matches.length} Matched</span>`;

      const matchStr = rec.matches.length > 0 ? rec.matches.join(", ") : "<em>None (Singleton)</em>";

      return `
        <tr>
          <td><strong style="font-family: var(--font-mono); color: var(--color-indigo);">${rec.entity_id}</strong></td>
          <td><strong>${escapeHtml(rec.business_name)}</strong></td>
          <td>${escapeHtml(rec.business_address)}</td>
          <td><span class="country-tag">${escapeHtml(rec.country)}</span></td>
          <td>${statusPill}</td>
          <td>${rec.candidates.length}</td>
          <td><span style="font-family: var(--font-mono); color: ${rec.matches.length > 0 ? 'var(--color-emerald)' : 'var(--text-subtle)'};">${matchStr}</span></td>
        </tr>
      `;
    }).join("");
  }

  // View Toggle (Card vs Table)
  viewCardsBtn.addEventListener("click", () => {
    viewCardsBtn.classList.add("active");
    viewTableBtn.classList.remove("active");
    recordsList.classList.remove("hidden");
    tableViewContainer.classList.add("hidden");
  });

  viewTableBtn.addEventListener("click", () => {
    viewTableBtn.classList.add("active");
    viewCardsBtn.classList.remove("active");
    recordsList.classList.add("hidden");
    tableViewContainer.classList.remove("hidden");
  });

  // Export Results CSV
  if (btnDownloadTableCsv) {
    btnDownloadTableCsv.addEventListener("click", () => {
      if (allRecords.length === 0) return;
      let csvContent = "source1_entity_id,business_name,address,country,is_singleton,matched_entity_ids\n";
      allRecords.forEach(r => {
        const matches = r.matches.join(";");
        const cleanName = `"${r.business_name.replace(/"/g, '""')}"`;
        const cleanAddr = `"${r.business_address.replace(/"/g, '""')}"`;
        csvContent += `${r.entity_id},${cleanName},${cleanAddr},${r.country},${r.is_singleton},${matches}\n`;
      });

      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", "entity_resolution_predictions.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Filter Listeners
  let debounceTimeout = null;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(loadRecords, 200);
  });

  countryFilter.addEventListener("change", loadRecords);
  typeFilter.addEventListener("change", renderFilteredRecords);

  // Initial Load
  fetchDatasetStatus();
  fetchStats();
  loadRecords();
});
