"use strict";

const latest = document.getElementById("current-edition");
if (latest?.hasAttribute("data-open") && !location.hash) location.replace(latest.href);

for (const input of document.querySelectorAll("input[data-filter]")) {
  const name = input.dataset.filter;
  const items = [...document.getElementById(name).querySelectorAll("[data-item]")];
  const status = document.querySelector(`[data-count="${name}"]`);
  input.addEventListener("input", () => {
    const terms = input.value.toLocaleLowerCase().trim().split(/\s+/);
    let shown = 0;
    for (const item of items) {
      const text = (item.textContent + " " + (item.dataset.search || "")).toLocaleLowerCase();
      item.hidden = !terms.every((term) => text.includes(term));
      if (!item.hidden) shown++;
    }
    if (status) {
      status.textContent = shown ? `${shown} matching entries` : "No matching entries";
      status.classList.toggle("sr-only", shown !== 0);
    }
  });
}
