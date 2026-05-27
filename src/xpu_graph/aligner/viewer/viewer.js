(function () {
  const payload = window.__ALIGNER_GRAPH__;
  if (!payload) {
    document.body.innerHTML = "<p>Missing graph payload. Expected graph-data.js to define window.__ALIGNER_GRAPH__.</p>";
    return;
  }

  const state = {
    currentStep: Array.isArray(payload.steps) && payload.steps.length ? payload.steps[0] : 0,
    focusedModuleId: {},
    hoveredModuleId: {},
    selected: null,
    linkedNodeIds: new Set(),
    expandedModules: {},
    stageFilters: {},
  };

  const stagePanels = new Map();
  const stageMap = new Map();
  const gradForwardToBackward = new Map();
  const gradBackwardToForward = new Map();

  payload.grad_links.forEach((link) => {
    gradForwardToBackward.set(link.src, link.dst);
    gradBackwardToForward.set(link.dst, link.src);
  });

  payload.stages.forEach((stage) => {
    stage.modulesById = new Map(stage.modules.map((module) => [module.module_id, module]));
    stage.nodesById = new Map(stage.nodes.map((node) => [node.node_id, node]));
    stage.edgesById = new Map(stage.edges.map((edge) => [edge.edge_id, edge]));
    stageMap.set(stage.stage, stage);
    state.focusedModuleId[stage.stage] = null;
    state.hoveredModuleId[stage.stage] = null;
    state.expandedModules[stage.stage] = new Set(stage.modules.map((module) => module.module_id));
    state.stageFilters[stage.stage] = {};
  });

  const graphSummary = document.getElementById("graph-summary");
  const stageGrid = document.getElementById("stage-grid");

  graphSummary.textContent =
    payload.graph_id +
    " · " +
    payload.variant_ids.length +
    " variants · " +
    payload.steps.length +
    " steps · " +
    payload.stages.reduce((count, stage) => count + stage.nodes.length, 0) +
    " nodes";
  document.title = payload.graph_id;

  renderLegend();
  buildStagePanels();
  renderAll();

  function renderLegend() {
    const legend = document.getElementById("variant-legend");
    legend.innerHTML = "";
    payload.variant_ids.forEach((variantId) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = payload.variant_palette[variantId];
      const text = document.createElement("span");
      text.textContent = variantId + (variantId === payload.gold_vid ? " (gold)" : "");
      item.appendChild(swatch);
      item.appendChild(text);
      legend.appendChild(item);
    });
  }

  function buildStagePanels() {
    stageGrid.innerHTML = "";
    payload.stages.forEach((stage) => {
      const card = document.createElement("section");
      card.className = "stage-card";

      const stageLabel = stage.stage === "FORWARD" ? "Forward" : "Backward";
      card.innerHTML =
        '<div class="stage-header">' +
        '  <div class="stage-toolbar">' +
        '    <div class="toolbar-group">' +
        '      <h2 class="stage-title">' + stageLabel + "</h2>" +
        "      <span class=\"focus-pill\">Focus: <strong class=\"focus-name\">All modules</strong></span>" +
        "    </div>" +
        '    <div class="toolbar-group">' +
        '      <label>Step <select class="step-select"></select></label>' +
        '      <button type="button" class="focus-reset">Reset Focus</button>' +
        "    </div>" +
        "  </div>" +
        '  <p class="muted stage-status">Click a module to focus its subgraph. Click a node or edge to pin details below.</p>' +
        "</div>" +
        '<div class="stage-body">' +
        '  <section class="dag-pane">' +
        '    <div class="pane-title pane-title-dag">Alignment DAG</div>' +
        '    <svg class="stage-svg dag-svg"></svg>' +
        '    <aside class="tree-pane">' +
        '      <div class="pane-title pane-title-tree">Module Tree</div>' +
        '      <svg class="stage-svg tree-svg"></svg>' +
        "    </aside>" +
        "  </section>" +
        "</div>";

      const stepSelect = card.querySelector(".step-select");
      payload.steps.forEach((step) => {
        const option = document.createElement("option");
        option.value = String(step);
        option.textContent = String(step);
        stepSelect.appendChild(option);
      });
      stepSelect.value = String(state.currentStep);
      stepSelect.addEventListener("change", function () {
        state.currentStep = Number(stepSelect.value);
        syncStepSelectors();
        renderAll();
      });

      card.querySelector(".focus-reset").addEventListener("click", function () {
        state.focusedModuleId[stage.stage] = null;
        state.hoveredModuleId[stage.stage] = null;
        renderAll();
      });

      const treeSvg = card.querySelector(".tree-svg");
      const dagSvg = card.querySelector(".dag-svg");
      const treeViewport = createSvgElement("g");
      treeViewport.setAttribute("class", "tree-viewport");
      treeSvg.appendChild(treeViewport);
      const dagViewport = createSvgElement("g");
      dagViewport.setAttribute("class", "dag-viewport");
      dagSvg.appendChild(dagViewport);

      const panel = {
        stage,
        card,
        stepSelect,
        focusName: card.querySelector(".focus-name"),
        statusLine: card.querySelector(".stage-status"),
        treeSvg,
        treeViewport,
        dagSvg,
        dagViewport,
        treeZoom: attachPanZoom(treeSvg, treeViewport, { x: 0, y: 0, scale: 1 }),
        dagZoom: attachPanZoom(dagSvg, dagViewport, { x: 28, y: 18, scale: 1 }),
      };

      stagePanels.set(stage.stage, panel);
      stageGrid.appendChild(card);
    });
  }

  function renderAll() {
    syncLinkedNodes();
    payload.stages.forEach((stage) => {
      renderStage(stage);
    });
  }

  function syncStepSelectors() {
    stagePanels.forEach((panel) => {
      panel.stepSelect.value = String(state.currentStep);
    });
  }

  function syncLinkedNodes() {
    state.linkedNodeIds = new Set();
    if (!state.selected || state.selected.kind !== "node") {
      return;
    }
    const nodeId = state.selected.id;
    if (state.selected.stage === "FORWARD" && gradForwardToBackward.has(nodeId)) {
      state.linkedNodeIds.add(gradForwardToBackward.get(nodeId));
    } else if (state.selected.stage === "BACKWARD" && gradBackwardToForward.has(nodeId)) {
      state.linkedNodeIds.add(gradBackwardToForward.get(nodeId));
    }
  }

  function renderStage(stage) {
    const panel = stagePanels.get(stage.stage);
    const focusedModule = state.focusedModuleId[stage.stage]
      ? stage.modulesById.get(state.focusedModuleId[stage.stage])
      : null;
    const hoveredModule = state.hoveredModuleId[stage.stage]
      ? stage.modulesById.get(state.hoveredModuleId[stage.stage])
      : null;

    panel.focusName.textContent = focusedModule ? focusedModule.display_path : "All modules";
    panel.statusLine.textContent = hoveredModule
      ? "Hovering " + hoveredModule.display_path + " · DAG highlight is scoped to this subtree."
      : focusedModule
      ? "Focused on " + focusedModule.display_path + " · Reset focus to restore the full DAG."
      : "Click a module to focus its subgraph. Click a node or edge to pin details below.";

    renderTree(stage, panel, focusedModule, hoveredModule);
    renderDag(stage, panel, focusedModule, hoveredModule);
  }

  function renderTree(stage, panel, focusedModule, hoveredModule) {
    const layout = computeTreeLayout(stage);
    const width = Math.max(300, layout.maxDepth * 180 + 220);
    const height = Math.max(260, layout.rows.length * 34 + 40);
    panel.treeSvg.setAttribute("viewBox", "0 0 " + width + " " + height);
    panel.treeViewport.innerHTML = "";

    panel.treeZoom.fitToBounds(width, height, 0.04);

    layout.links.forEach((link) => {
      const path = createSvgElement("path");
      path.setAttribute("class", "tree-link");
      const midX = link.parent.x + 26;
      path.setAttribute(
        "d",
        "M " +
          link.parent.x +
          " " +
          link.parent.y +
          " C " +
          midX +
          " " +
          link.parent.y +
          ", " +
          midX +
          " " +
          link.child.y +
          ", " +
          link.child.x +
          " " +
          link.child.y
      );
      panel.treeViewport.appendChild(path);
    });

    layout.rows.forEach((row) => {
      const module = row.module;
      const group = createSvgElement("g");
      const classes = ["tree-node"];
      if (focusedModule && focusedModule.module_id === module.module_id) {
        classes.push("is-focused");
      }
      if (hoveredModule && hoveredModule.module_id === module.module_id) {
        classes.push("is-hovered");
      }
      group.setAttribute("class", classes.join(" "));
      group.setAttribute("transform", "translate(" + row.x + " " + row.y + ")");
      group.style.cursor = "pointer";

      const circle = createSvgElement("circle");
      circle.setAttribute("r", focusedModule && focusedModule.module_id === module.module_id ? "8" : "6");
      circle.setAttribute("cx", "0");
      circle.setAttribute("cy", "0");
      circle.setAttribute("stroke", stage.stage_color);
      group.appendChild(circle);

      const title = createSvgElement("title");
      title.textContent = module.display_path + (module.class_name ? " [" + module.class_name + "]" : "");
      group.appendChild(title);

      const label = createSvgElement("text");
      label.setAttribute("x", "14");
      label.setAttribute("y", "-1");
      label.textContent = module.display_name;
      group.appendChild(label);

      const meta = createSvgElement("text");
      meta.setAttribute("class", "tree-meta");
      meta.setAttribute("x", "14");
      meta.setAttribute("y", "14");
      meta.textContent = module.descendant_node_ids.length + " nodes";
      group.appendChild(meta);

      group.addEventListener("mouseenter", function () {
        state.hoveredModuleId[stage.stage] = module.module_id;
        renderAll();
      });
      group.addEventListener("mouseleave", function () {
        state.hoveredModuleId[stage.stage] = null;
        renderAll();
      });
      group.addEventListener("click", function () {
        state.focusedModuleId[stage.stage] =
          state.focusedModuleId[stage.stage] === module.module_id ? null : module.module_id;
        state.selected = { kind: "module", stage: stage.stage, id: module.module_id };
        renderAll();
      });

      panel.treeViewport.appendChild(group);
    });
    panel.treeZoom.apply();
  }

  function renderDag(stage, panel, focusedModule, hoveredModule) {
    const visibleNodeIds = getVisibleNodeIds(stage, focusedModule);
    const highlightedNodeIds = hoveredModule ? new Set(hoveredModule.descendant_node_ids) : null;
    const visibleNodes = stage.nodes.filter((node) => visibleNodeIds.has(node.node_id));
    const visibleEdges = stage.edges.filter((edge) => visibleNodeIds.has(edge.src) && visibleNodeIds.has(edge.dst));
    const layout = computeDagLayout(visibleNodes, visibleEdges, stage);
    const moduleBoxes = computeModuleBoxes(stage, visibleNodeIds, layout);

    panel.dagViewport.innerHTML = "";

    if (!panel.dagZoom.initialized) {
      panel.dagZoom.reset();
    }

    if (!visibleNodes.length) {
      const empty = createSvgElement("text");
      empty.setAttribute("class", "empty-state");
      empty.setAttribute("x", "60");
      empty.setAttribute("y", "60");
      empty.textContent = "No nodes are visible for the current focus.";
      panel.dagViewport.appendChild(empty);
      panel.dagZoom.apply();
      return;
    }

    const moduleLayer = createSvgElement("g");
    const edgeLayer = createSvgElement("g");
    const edgeLabelLayer = createSvgElement("g");
    const nodeLayer = createSvgElement("g");

    moduleBoxes.forEach((box) => {
      const group = createSvgElement("g");
      const classes = ["module-cluster"];
      if (focusedModule && focusedModule.module_id === box.module.module_id) {
        classes.push("is-focused");
      }
      if (hoveredModule && hoveredModule.module_id === box.module.module_id) {
        classes.push("is-hovered");
      }
      group.setAttribute("class", classes.join(" "));

      const rect = createSvgElement("rect");
      rect.setAttribute("x", String(box.left));
      rect.setAttribute("y", String(box.top));
      rect.setAttribute("width", String(box.right - box.left));
      rect.setAttribute("height", String(box.bottom - box.top));
      rect.setAttribute("fill", hexToRgba(stage.stage_color, Math.min(0.06 + box.depth * 0.035, 0.16)));
      rect.setAttribute("stroke", stage.stage_color);
      group.appendChild(rect);

      const labelBg = createSvgElement("rect");
      labelBg.setAttribute("class", "module-cluster-label-bg");
      labelBg.setAttribute("x", String(box.left + 10));
      labelBg.setAttribute("y", String(box.top + 8));
      labelBg.setAttribute("width", String(Math.max(76, box.label.length * 7 + 18)));
      labelBg.setAttribute("height", "22");
      labelBg.setAttribute("fill", hexToRgba(stage.stage_color, 0.14));
      labelBg.setAttribute("stroke", "none");
      group.appendChild(labelBg);

      const label = createSvgElement("text");
      label.setAttribute("class", "module-cluster-label");
      label.setAttribute("x", String(box.left + 20));
      label.setAttribute("y", String(box.top + 23));
      label.textContent = box.label;
      group.appendChild(label);

      const title = createSvgElement("title");
      title.textContent = box.module.display_path;
      group.appendChild(title);

      moduleLayer.appendChild(group);
    });

    visibleEdges.forEach((edge) => {
      const src = layout.nodePositions.get(edge.src);
      const dst = layout.nodePositions.get(edge.dst);
      if (!src || !dst) {
        return;
      }

      const path = createSvgElement("path");
      const color = edge.variant_ids.length === 1 ? payload.variant_palette[edge.variant_ids[0]] : "#9E9E9E";
      path.setAttribute("class", "dag-link");
      if (highlightedNodeIds && !(highlightedNodeIds.has(edge.src) && highlightedNodeIds.has(edge.dst))) {
        path.classList.add("is-dimmed");
      }
      path.setAttribute("stroke", color);
      path.setAttribute("d", edgePath(src, dst));
      const title = createSvgElement("title");
      title.textContent = edge.label + " | variants: " + edge.variant_ids.join(", ");
      path.appendChild(title);
      path.style.cursor = "pointer";
      path.addEventListener("click", function () {
        state.selected = { kind: "edge", stage: stage.stage, id: edge.edge_id };
        renderAll();
      });
      edgeLayer.appendChild(path);

      if (visibleEdges.length <= 80) {
        const label = createSvgElement("text");
        label.setAttribute("class", "dag-link-label");
        label.setAttribute("x", String((src.x + dst.x) / 2));
        label.setAttribute("y", String((src.y + dst.y) / 2 - 6));
        label.textContent = edge.label;
        edgeLabelLayer.appendChild(label);
      }
    });

    visibleNodes.forEach((node) => {
      const position = layout.nodePositions.get(node.node_id);
      if (!position) {
        return;
      }

      const group = createSvgElement("g");
      const classes = ["dag-node"];
      const isSelected =
        state.selected &&
        state.selected.kind === "node" &&
        state.selected.stage === stage.stage &&
        state.selected.id === node.node_id;
      const isLinked = state.linkedNodeIds.has(node.node_id);
      const isHoveredDimmed =
        highlightedNodeIds && !highlightedNodeIds.has(node.node_id) && !isSelected && !isLinked;

      if (isSelected) {
        classes.push("is-selected");
      }
      if (isLinked) {
        classes.push("is-linked");
      }
      if (isHoveredDimmed) {
        classes.push("is-dimmed");
      }

      group.setAttribute("class", classes.join(" "));
      group.setAttribute("transform", "translate(" + position.left + " " + position.top + ")");
      group.style.cursor = "pointer";

      const rect = createSvgElement("rect");
      rect.setAttribute("width", String(layout.nodeWidth));
      rect.setAttribute("height", String(layout.nodeHeight));
      rect.setAttribute("fill", node.fillcolor);
      rect.setAttribute("stroke", stepStatusColor(stage.stage_color, getNodeStepStatus(node, state.currentStep)));
      group.appendChild(rect);

      const title = createSvgElement("title");
      title.textContent = node.label + " | " + node.module_id;
      group.appendChild(title);

      const titleText = createSvgElement("text");
      titleText.setAttribute("class", "node-title");
      titleText.setAttribute("x", "14");
      titleText.setAttribute("y", "23");
      titleText.textContent = node.node_id + ": " + shortLabel(shortOpName(node.op_name), 22);
      group.appendChild(titleText);

      const subtitleText = createSvgElement("text");
      subtitleText.setAttribute("class", "node-subtitle");
      subtitleText.setAttribute("x", "14");
      subtitleText.setAttribute("y", "43");
      subtitleText.textContent = "module: " + shortLabel(modulePathFor(stage, node.module_id), 30);
      group.appendChild(subtitleText);

      const badge = createSvgElement("circle");
      badge.setAttribute("cx", String(layout.nodeWidth - 16));
      badge.setAttribute("cy", "16");
      badge.setAttribute("r", "5");
      badge.setAttribute("fill", statusBadgeFill(getNodeStepStatus(node, state.currentStep)));
      group.appendChild(badge);

      group.addEventListener("click", function () {
        state.selected = { kind: "node", stage: stage.stage, id: node.node_id };
        renderAll();
      });

      nodeLayer.appendChild(group);
    });

    panel.dagViewport.appendChild(moduleLayer);
    panel.dagViewport.appendChild(edgeLayer);
    panel.dagViewport.appendChild(edgeLabelLayer);
    panel.dagViewport.appendChild(nodeLayer);
    panel.dagZoom.apply();
  }

  function renderKvGrid(entries) {
    let html = '<dl class="kv-grid">';
    entries.forEach((entry) => {
      html += "<dt>" + escapeHtml(entry[0]) + "</dt><dd>" + escapeHtml(entry[1]) + "</dd>";
    });
    html += "</dl>";
    return html;
  }

  function renderVariantTable(variants) {
    let html =
      '<table class="variant-table"><thead><tr><th>Variant</th><th>DType</th><th>XOR</th><th>Max Diff</th><th>Status</th></tr></thead><tbody>';
    variants.forEach((variant) => {
      const status = variant.status === "no data" ? "no data" : variant.closeto ? "pass" : "fail";
      html +=
        "<tr>" +
        "<td>" + escapeHtml(variant.variant) + (variant.variant === payload.gold_vid ? " (gold)" : "") + "</td>" +
        "<td>" + escapeHtml(variant.dtype || "n/a") + "</td>" +
        "<td>" + escapeHtml(variant.xorsum || "n/a") + "</td>" +
        "<td>" + escapeHtml(variant.max_diff == null ? "n/a" : String(variant.max_diff)) + "</td>" +
        '<td class="' +
        (status === "pass" ? "status-pass" : status === "fail" ? "status-fail" : "") +
        '">' +
        escapeHtml(status) +
        "</td>" +
        "</tr>";
    });
    html += "</tbody></table>";
    return html;
  }

  function renderOps(ops) {
    if (!ops.length) {
      return "<p class=\"muted\">No ops recorded.</p>";
    }
    return '<ol class="ops-list">' + ops.map((op) => "<li>" + escapeHtml(op) + "</li>").join("") + "</ol>";
  }

  function makeInfoCard(title, bodyHtml) {
    const card = document.createElement("section");
    card.className = "info-card";
    card.innerHTML = "<h3>" + title + "</h3>" + bodyHtml;
    return card;
  }

  function computeTreeLayout(stage) {
    const rows = [];
    const links = [];
    let order = 0;
    let maxDepth = 0;

    function walk(moduleId, depth, parentRow) {
      const module = stage.modulesById.get(moduleId);
      const row = {
        module: module,
        depth: depth,
        x: 20 + depth * 28,
        y: 24 + order * 34,
      };
      rows.push(row);
      maxDepth = Math.max(maxDepth, depth);
      order += 1;

      if (parentRow) {
        links.push({ parent: parentRow, child: row });
      }

      module.child_module_ids.forEach((childId) => {
        walk(childId, depth + 1, row);
      });
    }

    walk(stage.root_module_id, 0, null);
    return { rows: rows, links: links, maxDepth: maxDepth };
  }

  function computeDagLayout(nodes, edges, stage) {
    const nodeWidth = 238;
    const nodeHeight = 58;
    const rankGap = 92;
    const laneGap = 44;
    const nodePositions = new Map();
    const indegree = new Map();
    const outgoing = new Map();
    const incoming = new Map();
    const rank = new Map();
    const visibleIds = new Set(nodes.map((node) => node.node_id));

    nodes.forEach((node) => {
      indegree.set(node.node_id, 0);
      outgoing.set(node.node_id, []);
      incoming.set(node.node_id, []);
      rank.set(node.node_id, 0);
    });

    edges.forEach((edge) => {
      if (!visibleIds.has(edge.src) || !visibleIds.has(edge.dst)) {
        return;
      }
      indegree.set(edge.dst, (indegree.get(edge.dst) || 0) + 1);
      outgoing.get(edge.src).push(edge.dst);
      incoming.get(edge.dst).push(edge.src);
    });

    const queue = nodes
      .filter((node) => indegree.get(node.node_id) === 0)
      .sort(compareNodesForLayout);
    const ordered = [];

    while (queue.length) {
      const current = queue.shift();
      ordered.push(current.node_id);
      (outgoing.get(current.node_id) || []).forEach((dst) => {
        rank.set(dst, Math.max(rank.get(dst), rank.get(current.node_id) + 1));
        indegree.set(dst, indegree.get(dst) - 1);
        if (indegree.get(dst) === 0) {
          queue.push(stage.nodesById.get(dst));
          queue.sort(compareNodesForLayout);
        }
      });
    }

    nodes.forEach((node) => {
      if (!ordered.includes(node.node_id)) {
        ordered.push(node.node_id);
      }
    });

    const lanes = new Map();
    ordered.forEach((nodeId) => {
      const lane = rank.get(nodeId) || 0;
      if (!lanes.has(lane)) {
        lanes.set(lane, []);
      }
      lanes.get(lane).push(stage.nodesById.get(nodeId));
    });

    Array.from(lanes.keys())
      .sort((left, right) => left - right)
      .forEach((lane) => {
        lanes.get(lane).sort(compareNodesForLayout);
      });

    const sortedLanes = Array.from(lanes.keys()).sort((left, right) => left - right);
    const maxLane = sortedLanes.length ? sortedLanes[sortedLanes.length - 1] : 0;
    const maxLaneWidth = Math.max.apply(
      null,
      sortedLanes.map((lane) => lanes.get(lane).length)
    );
    const orderIndex = new Map();

    sortedLanes.forEach((lane) => {
      const laneNodes = lanes.get(lane);
      laneNodes.sort((left, right) => {
        const leftBarycenter = computeBarycenter(left.node_id, incoming, orderIndex);
        const rightBarycenter = computeBarycenter(right.node_id, incoming, orderIndex);
        if (leftBarycenter != null && rightBarycenter != null && leftBarycenter !== rightBarycenter) {
          return leftBarycenter - rightBarycenter;
        }
        if (leftBarycenter != null && rightBarycenter == null) {
          return -1;
        }
        if (leftBarycenter == null && rightBarycenter != null) {
          return 1;
        }
        return compareNodesForLayout(left, right);
      });

      const laneOffset = ((maxLaneWidth - laneNodes.length) * (nodeWidth + laneGap)) / 2;
      laneNodes.forEach((node, index) => {
          const left = laneOffset + index * (nodeWidth + laneGap);
          const laneIndex = stage.stage === "BACKWARD" ? maxLane - lane : lane;
          const top = laneIndex * (nodeHeight + rankGap);
          nodePositions.set(node.node_id, {
            left: left,
            top: top,
            x: left + nodeWidth / 2,
            y: top + nodeHeight / 2,
          });
          orderIndex.set(node.node_id, index);
        });
      });

    enforceSiblingModuleSpacing(stage, visibleIds, nodePositions, nodeWidth, nodeHeight);

    return {
      nodePositions: nodePositions,
      nodeWidth: nodeWidth,
      nodeHeight: nodeHeight,
    };
  }

  function computeBarycenter(nodeId, incoming, orderIndex) {
    const parents = incoming.get(nodeId) || [];
    const knownParents = parents.filter((parentId) => orderIndex.has(parentId));
    if (!knownParents.length) {
      return null;
    }
    const total = knownParents.reduce((sum, parentId) => sum + orderIndex.get(parentId), 0);
    return total / knownParents.length;
  }

  function enforceSiblingModuleSpacing(stage, visibleNodeIds, nodePositions, nodeWidth, nodeHeight) {
    const gap = 28;

    function shiftModule(moduleId, delta) {
      if (!delta) {
        return;
      }
      const module = stage.modulesById.get(moduleId);
      if (!module) {
        return;
      }
      module.descendant_node_ids.forEach((nodeId) => {
        if (!visibleNodeIds.has(nodeId) || !nodePositions.has(nodeId)) {
          return;
        }
        const pos = nodePositions.get(nodeId);
        pos.left += delta;
        pos.x += delta;
      });
    }

    function walk(moduleId) {
      const module = stage.modulesById.get(moduleId);
      if (!module) {
        return;
      }

      module.child_module_ids.forEach((childId) => {
        walk(childId);
      });

      const boxes = computeModuleBoxes(stage, visibleNodeIds, { nodePositions: nodePositions, nodeWidth: nodeWidth, nodeHeight: nodeHeight })
        .filter((box) => box.parentModuleId === moduleId)
        .sort((left, right) => left.left - right.left);

      for (let index = 1; index < boxes.length; index += 1) {
        const previous = boxes[index - 1];
        const current = boxes[index];
        const overlap = previous.right + gap - current.left;
        if (overlap > 0) {
          shiftModule(current.module.module_id, overlap);
          for (let next = index; next < boxes.length; next += 1) {
            boxes[next].left += overlap;
            boxes[next].right += overlap;
          }
        }
      }
    }

    walk(stage.root_module_id);
  }

  function computeModuleBoxes(stage, visibleNodeIds, layout) {
    const nodeBounds = new Map();
    visibleNodeIds.forEach((nodeId) => {
      const position = layout.nodePositions.get(nodeId);
      if (!position) {
        return;
      }
      nodeBounds.set(nodeId, {
        left: position.left,
        top: position.top,
        right: position.left + layout.nodeWidth,
        bottom: position.top + layout.nodeHeight,
      });
    });

    const boxes = [];

    function walk(moduleId, depth) {
      const module = stage.modulesById.get(moduleId);
      if (!module) {
        return null;
      }

      const childBoxes = [];
      module.child_module_ids.forEach((childId) => {
        const childBox = walk(childId, depth + 1);
        if (childBox) {
          childBoxes.push(childBox);
        }
      });

      const directBounds = module.direct_node_ids
        .filter((nodeId) => visibleNodeIds.has(nodeId))
        .map((nodeId) => nodeBounds.get(nodeId))
        .filter(Boolean);

      if (module.module_id === stage.root_module_id) {
        return null;
      }

      const allBounds = directBounds.concat(
        childBoxes.map((box) => ({
          left: box.left,
          top: box.top,
          right: box.right,
          bottom: box.bottom,
        }))
      );
      if (!allBounds.length) {
        return null;
      }

      const padX = Math.max(14, 22 - depth * 2);
      const padBottom = Math.max(14, 18 - depth);
      const labelPad = Math.max(28, 34 - depth * 2);
      const box = {
        module: module,
        label: shortLabel(module.display_name, 22),
        depth: depth,
        parentModuleId: module.parent_module_id,
        left: Math.min.apply(null, allBounds.map((item) => item.left)) - padX,
        top: Math.min.apply(null, allBounds.map((item) => item.top)) - labelPad,
        right: Math.max.apply(null, allBounds.map((item) => item.right)) + padX,
        bottom: Math.max.apply(null, allBounds.map((item) => item.bottom)) + padBottom,
      };
      boxes.push(box);
      return box;
    }

    walk(stage.root_module_id, 0);
    return boxes.sort((left, right) => left.depth - right.depth || left.left - right.left);
  }

  function compareNodesForLayout(left, right) {
    const leftModule = left.module_id || "";
    const rightModule = right.module_id || "";
    if (leftModule !== rightModule) {
      return leftModule.localeCompare(rightModule);
    }
    return left.node_id - right.node_id;
  }

  function getVisibleNodeIds(stage, focusedModule) {
    if (!focusedModule) {
      return new Set(stage.nodes.map((node) => node.node_id));
    }
    return new Set(focusedModule.descendant_node_ids);
  }

  function getNodeStepStatus(node, step) {
    const detail = node.step_details.find((candidate) => Number(candidate.step) === Number(step));
    if (!detail) {
      return "missing";
    }
    let seenData = false;
    let failing = false;
    detail.variants.forEach((variant) => {
      if (variant.status === "no data") {
        return;
      }
      seenData = true;
      if (variant.variant !== payload.gold_vid && variant.closeto === false) {
        failing = true;
      }
    });
    if (!seenData) {
      return "missing";
    }
    return failing ? "fail" : "pass";
  }

  function stepStatusColor(stageColor, status) {
    if (status === "fail") {
      return "#9d3528";
    }
    if (status === "pass") {
      return stageColor;
    }
    return "#9c9488";
  }

  function statusBadgeFill(status) {
    if (status === "fail") {
      return "#d36253";
    }
    if (status === "pass") {
      return "#2f8b64";
    }
    return "#b0a99f";
  }

  function shortLabel(text, maxLength) {
    return text.length <= maxLength ? text : text.slice(0, maxLength - 1) + "…";
  }

  function shortOpName(opName) {
    const pieces = opName.split(/[.:]/).filter(Boolean);
    return pieces.length ? pieces[pieces.length - 1] : opName;
  }

  function modulePathFor(stage, moduleId) {
    const module = stage.modulesById.get(moduleId);
    return module ? module.display_path : "root";
  }

  function visibleVariantsText(node) {
    return payload.variant_ids.filter((variantId) => node.variant_presence[variantId]).join(", ");
  }

  function edgePath(src, dst) {
    const startX = src.left + 119;
    const startY = src.top + 58;
    const endX = dst.left + 119;
    const endY = dst.top;
    const dy = Math.max(42, (endY - startY) / 2);
    return (
      "M " +
      startX +
      " " +
      startY +
      " C " +
      startX +
      " " +
      (startY + dy) +
      ", " +
      endX +
      " " +
      (endY - dy) +
      ", " +
      endX +
      " " +
      endY
    );
  }

  function attachPanZoom(svg, viewport, initialTransform) {
    const model = {
      scale: initialTransform && initialTransform.scale ? initialTransform.scale : 1,
      x: initialTransform && initialTransform.x != null ? initialTransform.x : 24,
      y: initialTransform && initialTransform.y != null ? initialTransform.y : 24,
      initialized: false,
      pointerId: null,
      dragOrigin: null,
    };

    function apply() {
      viewport.setAttribute("transform", "translate(" + model.x + " " + model.y + ") scale(" + model.scale + ")");
      api.initialized = model.initialized;
    }

    function reset() {
      model.scale = initialTransform && initialTransform.scale ? initialTransform.scale : 1;
      model.x = initialTransform && initialTransform.x != null ? initialTransform.x : 24;
      model.y = initialTransform && initialTransform.y != null ? initialTransform.y : 24;
      model.initialized = true;
      apply();
    }

    function fitToBounds(contentWidth, contentHeight, paddingRatio) {
      if (model.initialized || !contentWidth || !contentHeight) {
        return;
      }
      const rect = svg.getBoundingClientRect();
      const viewBox = svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal : null;
      const viewportWidth = viewBox && viewBox.width ? viewBox.width : rect.width;
      const viewportHeight = viewBox && viewBox.height ? viewBox.height : rect.height;
      if (!viewportWidth || !viewportHeight) {
        reset();
        return;
      }

      const padding = paddingRatio || 0;
      const scale = Math.max(0.01, Math.min(
        (viewportWidth * (1 - padding * 2)) / contentWidth,
        (viewportHeight * (1 - padding * 2)) / contentHeight
      ));
      model.scale = scale;
      model.x = (viewportWidth - contentWidth * scale) / 2;
      model.y = (viewportHeight - contentHeight * scale) / 2;
      model.initialized = true;
      apply();
    }

    svg.addEventListener(
      "wheel",
      function (event) {
        event.preventDefault();
        const bounds = svg.getBoundingClientRect();
        const originX = event.clientX - bounds.left;
        const originY = event.clientY - bounds.top;
        const direction = event.deltaY < 0 ? 1.12 : 0.9;
        const nextScale = Math.max(model.scale * direction, 0.01);
        const scaleDelta = nextScale / model.scale;
        model.x = originX - (originX - model.x) * scaleDelta;
        model.y = originY - (originY - model.y) * scaleDelta;
        model.scale = nextScale;
        model.initialized = true;
        apply();
      },
      { passive: false }
    );

    svg.addEventListener("pointerdown", function (event) {
      model.pointerId = event.pointerId;
      model.dragOrigin = { x: event.clientX, y: event.clientY, tx: model.x, ty: model.y };
      svg.setPointerCapture(event.pointerId);
    });

    svg.addEventListener("pointermove", function (event) {
      if (model.pointerId !== event.pointerId || !model.dragOrigin) {
        return;
      }
      model.x = model.dragOrigin.tx + (event.clientX - model.dragOrigin.x);
      model.y = model.dragOrigin.ty + (event.clientY - model.dragOrigin.y);
      model.initialized = true;
      apply();
    });

    function endDrag(event) {
      if (model.pointerId === event.pointerId) {
        model.pointerId = null;
        model.dragOrigin = null;
      }
    }

    svg.addEventListener("pointerup", endDrag);
    svg.addEventListener("pointercancel", endDrag);

    const api = { apply: apply, reset: reset, fitToBounds: fitToBounds, initialized: false };
    return api;
  }

  function createSvgElement(name) {
    return document.createElementNS("http://www.w3.org/2000/svg", name);
  }
  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function hexToRgba(hex, alpha) {
    const value = hex.replace("#", "");
    const normalized = value.length === 3
      ? value.split("").map((part) => part + part).join("")
      : value;
    const red = parseInt(normalized.slice(0, 2), 16);
    const green = parseInt(normalized.slice(2, 4), 16);
    const blue = parseInt(normalized.slice(4, 6), 16);
    return "rgba(" + red + ", " + green + ", " + blue + ", " + alpha + ")";
  }
})();
