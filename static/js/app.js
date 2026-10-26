document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const loadSampleBtn = document.getElementById("load-sample-btn");
    const fileInfoBar = document.getElementById("file-info-bar");
    const fileNameDisplay = document.getElementById("file-name-display");
    const fileRowsDisplay = document.getElementById("file-rows-display");
    const fileColsDisplay = document.getElementById("file-cols-display");
    const fileMissingDisplay = document.getElementById("file-missing-display");
    const changeFileBtn = document.getElementById("change-file-btn");

    const targetColSelect = document.getElementById("target-col-select");
    const scalingSelect = document.getElementById("scaling-select");
    const encodingSelect = document.getElementById("encoding-select");
    const outlierSelect = document.getElementById("outlier-select");
    const dropDuplicatesCheck = document.getElementById("drop-duplicates-check");
    const dropConstantCheck = document.getElementById("drop-constant-check");
    const extractDatetimeCheck = document.getElementById("extract-datetime-check");

    const processBtn = document.getElementById("process-btn");
    const loadingState = document.getElementById("loading-state");
    const resultsSection = document.getElementById("results-section");

    // Results elements
    const executionTimeBadge = document.getElementById("execution-time-badge");
    const statInputShape = document.getElementById("stat-input-shape");
    const statInputMem = document.getElementById("stat-input-mem");
    const statOutputShape = document.getElementById("stat-output-shape");
    const statOutputMem = document.getElementById("stat-output-mem");
    const statDuplicates = document.getElementById("stat-duplicates");
    const statOutliers = document.getElementById("stat-outliers");

    const tagsPruned = document.getElementById("tags-pruned");
    const tagsEncoded = document.getElementById("tags-encoded");
    const tagsDatetime = document.getElementById("tags-datetime");

    const cleanedTable = document.getElementById("cleaned-table");
    const rawTable = document.getElementById("raw-table");
    const auditLogText = document.getElementById("audit-log-text");

    const downloadCsvBtn = document.getElementById("download-csv-btn");
    const downloadReportBtn = document.getElementById("download-report-btn");
    const downloadModelBtn = document.getElementById("download-model-btn");

    // Application state
    let currentDataset = null;

    // --- Drag & Drop Listeners ---
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    changeFileBtn.addEventListener("click", () => {
        resetState();
    });

    loadSampleBtn.addEventListener("click", async () => {
        try {
            loadSampleBtn.disabled = true;
            loadSampleBtn.textContent = "⏳ Generating sample...";
            const res = await fetch("/api/load-sample", { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                onDatasetLoaded(data);
            } else {
                alert(`Error loading sample: ${data.message}`);
            }
        } catch (err) {
            alert(`Network error: ${err.message}`);
        } finally {
            loadSampleBtn.disabled = false;
            loadSampleBtn.textContent = "⚡ Load Sample Messy Dataset";
        }
    });

    // --- Upload Handler ---
    async function handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith(".csv")) {
            alert("Please upload a .csv file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            dropZone.querySelector("h3").textContent = "Uploading & inspecting...";
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            if (data.status === "success") {
                onDatasetLoaded(data);
            } else {
                alert(`Upload failed: ${data.message}`);
                dropZone.querySelector("h3").textContent = "Drag & Drop your CSV file here";
            }
        } catch (err) {
            alert(`Error uploading file: ${err.message}`);
            dropZone.querySelector("h3").textContent = "Drag & Drop your CSV file here";
        }
    }

    function onDatasetLoaded(data) {
        currentDataset = data;

        // Update info bar
        fileNameDisplay.textContent = data.filename;
        fileRowsDisplay.textContent = `${data.rows.toLocaleString()} rows`;
        fileColsDisplay.textContent = `${data.columns.length} columns`;
        fileMissingDisplay.textContent = `${data.missing_total.toLocaleString()} missing values`;

        // Populate Target Dropdown
        targetColSelect.innerHTML = '<option value="">-- No Target (Unsupervised / All Features) --</option>';
        data.columns.forEach(col => {
            const opt = document.createElement("option");
            opt.value = col;
            opt.textContent = col;
            // auto select target if named target/churn/label/y
            if (["churn", "target", "label", "price", "y"].includes(col.toLowerCase())) {
                opt.selected = true;
            }
            targetColSelect.appendChild(opt);
        });

        // Populate raw data table preview
        renderTable(rawTable, data.preview, data.columns);

        // Transition UI
        dropZone.classList.add("hidden");
        fileInfoBar.classList.remove("hidden");
        processBtn.disabled = false;
        resultsSection.classList.add("hidden");
    }

    function resetState() {
        currentDataset = null;
        fileInput.value = "";
        dropZone.classList.remove("hidden");
        dropZone.querySelector("h3").textContent = "Drag & Drop your CSV file here";
        fileInfoBar.classList.add("hidden");
        processBtn.disabled = true;
        resultsSection.classList.add("hidden");
    }

    // --- Process Trigger Handler ---
    processBtn.addEventListener("click", async () => {
        if (!currentDataset) return;

        // Parse outlier options
        const outlierValue = outlierSelect.value;
        let outlierMethod = "iqr";
        let outlierAction = "clip";
        if (outlierValue === "none") {
            outlierMethod = "none";
            outlierAction = "none";
        } else if (outlierValue === "zscore-clip") {
            outlierMethod = "zscore";
            outlierAction = "clip";
        } else if (outlierValue === "iqr-drop") {
            outlierMethod = "iqr";
            outlierAction = "drop";
        }

        const payload = {
            task_id: currentDataset.task_id,
            file_path: currentDataset.file_path,
            target_column: targetColSelect.value,
            scaling_strategy: scalingSelect.value,
            encoding_strategy: encodingSelect.value,
            outlier_method: outlierMethod,
            outlier_action: outlierAction,
            drop_duplicates: dropDuplicatesCheck.checked,
            drop_constant_columns: dropConstantCheck.checked,
            extract_datetime_features: extractDatetimeCheck.checked,
        };

        try {
            // UI state
            loadingState.classList.remove("hidden");
            resultsSection.classList.add("hidden");
            processBtn.disabled = true;

            const res = await fetch("/api/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await res.json();
            if (data.status === "success") {
                displayResults(data);
            } else {
                alert(`Preprocessing failed: ${data.message}`);
            }
        } catch (err) {
            alert(`Error running pipeline: ${err.message}`);
        } finally {
            loadingState.classList.add("hidden");
            processBtn.disabled = false;
        }
    });

    // --- Render Preprocessing Results ---
    function displayResults(data) {
        const report = data.report;

        // Metric Badges
        executionTimeBadge.textContent = `Completed in ${report.execution_time_seconds} seconds`;
        statInputShape.textContent = `${report.input_rows} x ${report.input_columns}`;
        statInputMem.textContent = `${report.input_memory_mb || 0} MB`;
        
        statOutputShape.textContent = `${report.output_rows} x ${report.output_columns}`;
        statOutputMem.textContent = `${report.output_memory_mb || 0} MB`;

        statDuplicates.textContent = report.duplicates_removed || 0;

        const totalOutliers = report.outliers_treated ? Object.values(report.outliers_treated).reduce((a, b) => a + b, 0) : 0;
        statOutliers.textContent = totalOutliers;

        // Pruned columns tags
        const droppedMap = report.dropped_columns || {};
        const allDropped = [
            ...(droppedMap.constant_columns || []),
            ...(droppedMap.high_missing_columns || []),
            ...(droppedMap.id_columns || []),
        ];
        renderTags(tagsPruned, allDropped, "tag-drop");

        // Encoded tags
        const encMap = report.encoded_columns || {};
        const allEncoded = [
            ...(encMap.one_hot || []).map(c => `${c} (One-Hot)`),
            ...(encMap.label_encoded || []).map(c => `${c} (Label)`),
            ...(encMap.frequency_encoded || []).map(c => `${c} (Frequency)`),
        ];
        renderTags(tagsEncoded, allEncoded, "tag-enc");

        // Datetime tags
        renderTags(tagsDatetime, report.extracted_datetime_features || [], "tag-dt");

        // Render Table Previews
        renderTable(cleanedTable, data.preview, data.output_columns);

        // Audit Log text
        auditLogText.textContent = data.summary_text;

        // Download Links
        downloadCsvBtn.href = data.download_csv_url;
        downloadReportBtn.href = data.download_report_url;
        downloadModelBtn.href = data.download_pipeline_url;

        // Show Results
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    function renderTags(container, items, tagClass) {
        container.innerHTML = "";
        if (!items || items.length === 0) {
            container.innerHTML = '<span class="text-muted">None</span>';
            return;
        }
        items.forEach(item => {
            const span = document.createElement("span");
            span.className = `tag ${tagClass}`;
            span.textContent = item;
            container.appendChild(span);
        });
    }

    function renderTable(tableElem, records, columns) {
        if (!records || records.length === 0) {
            tableElem.innerHTML = "<tr><td class='text-muted'>No preview data available</td></tr>";
            return;
        }

        const thead = tableElem.querySelector("thead");
        const tbody = tableElem.querySelector("tbody");

        thead.innerHTML = "";
        tbody.innerHTML = "";

        // Headers
        const headerRow = document.createElement("tr");
        columns.forEach(col => {
            const th = document.createElement("th");
            th.textContent = col;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        // Rows
        records.forEach(row => {
            const tr = document.createElement("tr");
            columns.forEach(col => {
                const td = document.createElement("td");
                const val = row[col];
                td.textContent = (val !== null && val !== undefined) ? val : "";
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    // --- Tab Switching ---
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(btn.getAttribute("data-tab"));
            if (targetPane) targetPane.classList.add("active");
        });
    });

});
