// The visualizer app: sign-in, then pages chosen by the URL.
//
//   /                          runs (everyone's, newest first)
//   /runs/<run>                a run's episodes
//   /runs/<run>/episodes/<n>   an episode: video, boxes and signals

import "uplot/dist/uPlot.min.css";
import "./style.css";
import { api, Me, SignInNeeded } from "./api";
import * as auth from "./auth";
import { h, link, setRouter } from "./dom";
import { episodePage } from "./episode";
import { runPage, runsPage } from "./runs";

const app = document.getElementById("app")!;
let me: Me | null = null;
// Each page can leave something to stop when it's left (e.g. a playing video).
let cleanup: (() => void) | null = null;

function header(): HTMLElement {
  return h("header", null,
    link("/", h("strong", null, "Wildhat"), " visualizer"),
    me && h("span", { class: "who" }, me.handle, " · ",
      h("button", { class: "linkish", onclick: () => { auth.signOut(); me = null; route(); } }, "Sign out")),
  );
}

function show(...content: Node[]): void {
  app.replaceChildren(header(), h("main", null, ...content));
}

function signInPage(message = ""): void {
  const email = h("input", { type: "email", placeholder: "you@school.org", required: true, autocomplete: "email" });
  const status = h("p", { class: "muted" }, message);
  const form = h("form", {
    class: "signin",
    onsubmit: async (e: Event) => {
      e.preventDefault();
      status.textContent = "Sending...";
      try {
        await auth.sendLink(email.value.trim());
        status.textContent = `Check ${email.value.trim()} for a sign-in link, and open it in this browser.`;
      } catch (err) {
        status.textContent = (err as Error).message;
      }
    },
  },
    h("h1", null, "Sign in"),
    h("p", null, "With the email address on the team roster, the same as ", h("code", null, "wildhat login"), "."),
    h("div", { class: "row" }, email, h("button", { type: "submit" }, "Send me a link")),
    status,
  );
  show(form);
  email.focus();
}

async function route(): Promise<void> {
  cleanup?.();
  cleanup = null;
  if (location.pathname === "/signin") {
    try {
      const next = await auth.finishFromLink();
      history.replaceState(null, "", next ?? "/");
    } catch (err) {
      history.replaceState(null, "", "/");
      return signInPage((err as Error).message);
    }
  }
  if (!auth.signedIn()) return signInPage();
  try {
    me ??= await api.me();
    const parts = location.pathname.split("/").filter(Boolean).map(decodeURIComponent);
    if (parts.length === 0) {
      show(await runsPage(me));
    } else if (parts[0] === "runs" && parts.length === 2) {
      show(await runPage(parts[1]));
    } else if (parts[0] === "runs" && parts[2] === "episodes" && parts.length === 4) {
      const page = await episodePage(parts[1], Number(parts[3]));
      cleanup = page.stop;
      show(page.element);
    } else {
      show(h("p", null, "There's no such page. ", link("/", "See all runs.")));
    }
  } catch (err) {
    if (err instanceof SignInNeeded) {
      auth.signOut();
      me = null;
      return signInPage("Please sign in again.");
    }
    show(h("p", { class: "error" }, (err as Error).message), h("p", null, link("/", "Back to runs")));
  }
}

async function start(): Promise<void> {
  const config = await api.config();
  auth.configure(config.firebase_api_key);
  setRouter(route);
  await route();
}

start().catch((err) => {
  app.replaceChildren(h("p", { class: "error" }, `The visualizer couldn't start: ${(err as Error).message}`));
});

