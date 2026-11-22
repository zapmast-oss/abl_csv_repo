"use strict";

const $ = (id) => document.getElementById(id);
const data = window.ABL_STATUS;

function element(tag, text, classes) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (classes) node.className = classes;
  return node;
}

function anchor(item, classes, label) {
  if (!item?.href) return element("span", label || item?.label || "", "muted");
  const node = element("a", label || item.label, classes);
  node.href = item.href;
  return node;
}

function links(items, classes = "link-row") {
  const node = element("div", "", classes);
  node.append(...items.filter(Boolean).map((item) => anchor(item)));
  return node;
}

const dayFormat = new Intl.DateTimeFormat("en-US", {
  day: "numeric", month: "long", year: "numeric", timeZone: "UTC",
});
const timeFormat = new Intl.DateTimeFormat("en-US", {
  day: "numeric", month: "short", hour: "numeric", minute: "2-digit",
  timeZone: "America/Los_Angeles", timeZoneName: "short",
});
const asDate = (value) => value ? dayFormat.format(new Date(value + "T00:00:00Z")) : "Unknown";
const asTime = (value) => value ? timeFormat.format(new Date(value)) : "No recorded run";

function badge(status) {
  const tones = {
    Operating: "good", Prepared: "good", Ready: "good", Succeeded: "good",
    "Verified capture": "good", Available: "good", Published: "good", Approved: "good",
    "Approved for manual release": "good", Running: "running",
    "Requires approval": "warn", "Requires review": "warn", Unconfirmed: "warn",
    Unavailable: "warn", "Different capture": "warn", "Drive dependent": "warn", Failed: "fail",
  };
  return element("span", status, "badge " + (tones[status] || ""));
}

function row(label, value) {
  const node = element("div", "", "row");
  node.append(element("strong", label), value);
  return node;
}

function stat(label, value, detail, href) {
  const node = element("div", "", "stat");
  const number = href ? anchor({href, label: value}, "stat-value") : element("span", value, "stat-value");
  node.append(element("div", label, "stat-label"), number);
  if (detail) node.append(element("span", detail, "stat-detail"));
  return node;
}

function renderBaseball() {
  const {baseball, operations, releases, warnings} = data;
  const day = baseball.cutoff_day;
  const sessions = [operations.session, ...operations.sessions].filter(Boolean);
  const failed = sessions.flatMap((session) =>
    session.stages.filter((stage) => stage.status === "Failed").map((stage) => ({
      title: stage.label + " failed", detail: stage.message, link: stage.log,
    })),
  );
  const pollFailure = operations.poll.status === "failed";
  const failures = failed.length + Number(pollFailure) + Number(operations.mirror.status === "Failed");
  const approvals = releases.filter((item) => !item.approval).length;
  $("league-date").textContent = baseball.league_date ? asDate(baseball.league_date) : "League date unavailable";
  $("date-source").textContent = baseball.date_source || "Open the latest verified OOTP capture";
  $("headline-stats").append(
    stat("Completed-game cutoff", asDate(baseball.cutoff),
      day ? day.completed + " of " + day.scheduled + " scheduled finals" : "", baseball.verified?.href),
    stat("Last production run", asTime(baseball.last_run.at), baseball.last_run.label),
    stat("Pending approvals", String(approvals), "External releases", "#archives"),
    stat("Failed tasks", String(failures), "See task records", "#operations"),
  );
  const protection = operations.h_ready ? [] : [{
    title: "League backup drive unavailable",
    detail: "Connect H: to complete the mandatory safety copy and close a protected session.",
  }];
  for (const item of [...protection, ...warnings, ...failed].slice(0, 4)) {
    const notice = element("article", "", "notice");
    notice.append(element("strong", item.title), element("p", item.detail));
    if (item.link) notice.append(anchor(item.link));
    $("alerts").append(notice);
  }
  const slate = baseball.slate;
  const window = $("playing-window");
  if (baseball.window) window.append(element("p", baseball.window));
  if (slate.games?.length) {
    window.append(element("p", asDate(slate.date) + " • " + slate.games.length + " scheduled games pending", "muted"));
    const matchups = element("div", "", "schedule");
    for (const game of slate.games) {
      const node = element("span", game.away + " at " + game.home, "matchup");
      node.title = "OOTP game " + game.id;
      matchups.append(node);
    }
    window.append(matchups);
  } else {
    window.append(element("p", "No upcoming regular-season slate in the verified capture.", "muted"));
  }
  for (const item of baseball.exports) {
    const node = element("div");
    node.append(row(item.label, badge(item.status)), element("p", item.detail, "body-note"), links([item.link]));
    $("source-state").append(node);
  }
  if (!baseball.exports.length) $("source-state").append(element("p", "CSV comparison unavailable.", "muted"));
}

function renderDesk() {
  const desk = data.desk;
  if (desk.cutoff) $("desk-cutoff").textContent = "Coverage through " + asDate(desk.cutoff);
  $("desk-links").append(...desk.links.map((item) => anchor(item, "tile-link")));
  if (!desk.links.length) $("desk-links").append(element("p", "No current newsroom output is available.", "muted"));
  if (desk.reviewed.length) {
    $("desk-review").append(element("p", desk.reviewed.length + " reviewed articles in the local issue.", "body-note"));
  }
}

function renderDivisions() {
  for (const division of data.projects.filter((item) => item.id.startsWith("division-"))) {
    const panel = element("article", "", "panel division " + (division.id.includes("-abc-") ? "abc" : ""));
    panel.append(element("h3", division.title));
    if (division.movement) panel.append(element("p", division.movement, "muted"));
    const teams = element("ul", "", "teams");
    for (const team of data.projects.filter((item) => item.division === division.id)) {
      const entry = element("li", "", "row");
      entry.append(anchor(team.report, "", team.title), element("span", team.record || "—", "record"));
      teams.append(entry);
    }
    panel.append(teams);
    $("divisions").append(panel);
  }
}

function projectCard(project) {
  const panel = element("article", "", "panel program");
  panel.append(element("h3", project.title), badge(project.status), element("p", project.purpose),
    element("p", project.next_action, "body-note"), links(project.evidence.filter((source) => source.href)));
  return panel;
}

function stageRow(stage) {
  const node = element("div", "", "stage");
  const state = element("div");
  state.append(badge(stage.status));
  const body = element("div");
  if (stage.message) body.append(element("p", stage.message, "body-note"));
  const when = stage.at ? asTime(stage.at) : "";
  if (when) body.append(element("p", [when, stage.source_cutoff && "Cutoff " + stage.source_cutoff].filter(Boolean).join(" • "), "stat-detail"));
  const references = [stage.log, ...stage.outputs].filter(Boolean);
  if (references.length) body.append(links(references));
  node.append(element("strong", stage.label), state, body);
  return node;
}

function renderOperations() {
  const ops = data.operations;
  const modes = [
    ["Quick Play", "Play, save, preserve the league, and close the checkpoint."],
    ["Standard Session", "Add a verified CSV capture, repository Preview and Update, and the local Desk when useful."],
    ["Featured Production", "Add a recording and master backup, then track approved releases and their receipts."],
  ];
  for (const [title, summary] of modes) {
    const panel = element("article", "", "panel mode");
    panel.append(element("h3", title), element("p", summary));
    $("modes").append(panel);
  }
  $("operation-links").append(...[data.runbook, data.verification].filter(Boolean).map((item) => anchor(item)));
  const drives = element("div");
  drives.append(row("H: league safety copy", badge(ops.h_ready ? "Available" : "Unavailable")),
    row("I: mirror", badge(ops.i_ready ? "Available" : "Unavailable")));
  const poll = element("div");
  const pollState = {failed: "Failed", unchanged: "Verified capture", success: "Succeeded"};
  poll.append(row("Newsroom poll", badge(pollState[ops.poll.status] || "No receipt")),
    element("p", "Last check " + asTime(ops.poll.last_check_utc), "body-note"),
    links([ops.poll_log, ops.input_log]));
  $("protection").append(drives, poll);
  const panel = element("article", "", "panel session-panel");
  const session = ops.session;
  if (session) {
    panel.append(row(session.mode + " · " + session.trigger, badge(session.closed_at ? "Closed" : "Open")),
      element("p", session.window + " • " + session.id, "body-note"), links([session.record, session.checkpoint]));
    for (const stage of session.stages) panel.append(stageRow(stage));
  } else {
    panel.append(element("h3", "Session history"), element("p", "No Command Center sessions recorded. Open Production Control after saving in OOTP.", "body-note"));
  }
  $("session").append(panel);
  if (ops.sessions.length) {
    const carry = element("article", "", "panel carryover");
    carry.append(element("h3", "Earlier sessions and deferred work"));
    for (const previous of ops.sessions) {
      const tasks = previous.stages.filter((stage) => ["Failed", "Deferred", "Requires approval", "Running"].includes(stage.status));
      const item = element("details");
      item.append(element("summary", previous.mode + " · " + asDate(previous.league_date) +
        " · " + (previous.closed_at ? "Closed" : "Open") + " · " + tasks.length + " pending"));
      const body = element("div");
      body.append(element("p", previous.window + " • " + previous.id, "body-note"), links([previous.record, previous.checkpoint]));
      body.append(...tasks.map(stageRow));
      item.append(body);
      carry.append(item);
    }
    $("deferred").append(carry);
  }
}

function renderArchives() {
  const {releases, operations: ops} = data;
  if (!releases.length) {
    $("releases").append(element("p", "No external releases are queued. Select an artifact in Production Control to request approval.", "muted"));
  }
  for (const release of releases) {
    const panel = element("div", "", "release");
    let status = "Requires approval";
    if (release.receipt) status = "Published";
    else if (release.approval) status = "Approved for manual release";
    panel.append(row(release.title, badge(status)),
      element("p", [release.channel, release.lane, release.source_cutoff, release.id].filter(Boolean).join(" • "), "body-note"));
    if (release.approval) {
      panel.append(element("p", "Approved by " + release.approval.reviewer + " • " + asTime(release.approval.at), "body-note"));
    }
    const receipt = release.receipt ? {href: release.receipt.url, label: "Recorded public URL"} : null;
    panel.append(links([release.file, receipt]));
    $("releases").append(panel);
  }
  const archive = $("archive-lanes");
  const html = element("article", "", "panel program");
  html.append(element("h3", "OOTP HTML reports"),
    element("p", ops.html_at ? "Local index generated " + asTime(ops.html_at) + "." : "Local report index unavailable."),
    links([ops.html]));
  archive.append(html);
  const website = data.projects.find((item) => item.id === "html-website");
  if (website) archive.append(projectCard(website));
  const mirror = element("article", "", "panel program");
  mirror.append(element("h3", "Read-only H: to I: verification"), badge(ops.mirror.status),
    element("p", ops.mirror.detail), element("p", "Last receipt " + asTime(ops.mirror.at), "body-note"), links([ops.mirror.link]));
  archive.append(mirror);
  for (const id of ["statsplus-update", "almanac", "session-archives"]) {
    const project = data.projects.find((item) => item.id === id);
    if (project) archive.append(projectCard(project));
  }
}

function renderRegistry() {
  const view = $("project-view").value;
  const query = $("project-search").value.trim().toLocaleLowerCase();
  const projects = data.projects.filter((project) => {
    const match = [project.title, project.status, project.owner, project.priority, project.purpose, project.next_action]
      .some((field) => field.toLocaleLowerCase().includes(query));
    return match && (view === "all" || project.basis === view);
  });
  $("project-count").textContent = projects.length + " of " + data.projects.length + " projects";
  const list = $("projects");
  list.className = "projects";
  list.replaceChildren();
  for (const project of projects) {
    const item = element("details");
    const summary = element("summary");
    const title = element("span", "", "project-summary");
    title.append(element("strong", project.title), badge(project.status));
    summary.append(title);
    const body = element("div");
    body.append(element("p", project.purpose));
    const metadata = element("div", "", "project-metadata");
    metadata.append(...[project.basis, project.priority, project.owner, project.automation,
      project.last_verified && "Verified " + asDate(project.last_verified)].filter(Boolean).map((text) => element("span", text)));
    body.append(metadata, element("p", project.next_action));
    if (project.dependencies.length) {
      body.append(element("p", "Depends on " + project.dependencies.join(", "), "body-note"));
    }
    const evidence = element("div", "", "project-evidence");
    evidence.append(...project.evidence.map((source) => anchor(source)));
    body.append(evidence);
    item.append(summary, body);
    list.append(item);
  }
}

function render() {
  if (!data) {
    $("unavailable").hidden = false;
    return;
  }
  $("dashboard").hidden = false;
  $("refreshed").textContent = "Status refreshed " + asTime(data.generated_at);
  $("references").append(...[data.runbook, data.registry, data.verification].filter(Boolean).map((item) => anchor(item)));
  renderBaseball();
  renderDesk();
  renderDivisions();
  $("programs").append(...data.projects.filter((project) => project.section === "programming").map(projectCard));
  renderOperations();
  renderArchives();
  $("registry-size").textContent = data.projects.length + " projects";
  renderRegistry();
  $("project-view").addEventListener("change", renderRegistry);
  $("project-search").addEventListener("input", renderRegistry);
  const updateNavigation = () => {
    const current = location.hash || "#baseball";
    for (const item of document.querySelectorAll(".sidebar nav a")) {
      if (item.getAttribute("href") === current) item.setAttribute("aria-current", "location");
      else item.removeAttribute("aria-current");
    }
  };
  window.addEventListener("hashchange", updateNavigation);
  updateNavigation();
}

render();
