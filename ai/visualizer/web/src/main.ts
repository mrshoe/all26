// The visualizer app: sign-in, then pages chosen by the URL.
//
//   /                          runs (everyone's, newest first)
//   /runs/<run>                a run: its episodes, and the first one
//   /runs/<run>/episodes/<n>   the same, showing episode n: video, boxes
//                              and signals
//
// Choosing another episode of the same run only changes the main pane.

import "uplot/dist/uPlot.min.css";
import "./style.css";
import { api, Me, SignInNeeded } from "./api";
import * as auth from "./auth";
import logo from "./assets/wildhats-logo.png";
import { h, icon, link, setRouter } from "./dom";
import { RunView, runView, runsPage } from "./runs";

const app = document.getElementById("app")!;
let me: Me | null = null;
// Each page can leave something to stop when it's left (e.g. a playing video).
let cleanup: (() => void) | null = null;
// The run page showing, so changing episode keeps it.
let shown: { id: string; view: RunView } | null = null;

function header(): HTMLElement {
  const home = link("/", h("img", { class: "header__logo", src: logo, alt: "" }),
    h("span", { class: "header__title" }, "Team 100 ", h("span", null, "Visualizer")));
  home.className = "header__home";
  return h("header", { class: "header" },
    home,
    h("span", { class: "header__spacer" }),
    me && h("span", { class: "header__me" }, me.handle),
    me && h("button", {
      class: "btn btn--sm",
      onclick: () => { auth.signOut(); me = null; route(); },
    }, icon("signout"), "Sign out"),
  );
}

function show(...content: Node[]): void {
  app.replaceChildren(header(), h("main", { class: "main" }, ...content));
}

function leave(): void {
  cleanup?.();
  cleanup = null;
  shown = null;
}

function signInPage(message = ""): void {
  const email = h("input", { id: "si-email", type: "email", placeholder: "you@school.org", required: true, autocomplete: "email" });
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
    h("div", { class: "field" }, h("label", { for: "si-email" }, "Email"), email),
    h("button", { type: "submit", class: "btn btn--primary" }, "Send me a link"),
    status,
  );
  show(h("div", { class: "center" },
    h("img", { src: logo, alt: "" }),
    h("div", null,
      h("h1", { class: "center__title" }, "Team 100 Visualizer"),
      h("p", { class: "muted" }, "Sign in with the email address on the team roster, the same as ",
        h("code", null, "wildhat login"), "."),
    ),
    form,
  ));
  email.focus();
}

async function route(): Promise<void> {
  if (location.pathname === "/signin") {
    try {
      const next = await auth.finishFromLink();
      history.replaceState(null, "", next ?? "/");
    } catch (err) {
      history.replaceState(null, "", "/");
      return signInPage((err as Error).message);
    }
  }
  if (!auth.signedIn()) {
    leave();
    return signInPage();
  }
  try {
    me ??= await api.me();
    const parts = location.pathname.split("/").filter(Boolean).map(decodeURIComponent);
    const isRun = parts[0] === "runs" && (parts.length === 2 || (parts[2] === "episodes" && parts.length === 4));
    if (isRun) {
      const episode = parts.length === 4 ? Number(parts[3]) : null;
      if (shown?.id !== parts[1]) {
        leave();
        const view = await runView(parts[1]);
        shown = { id: parts[1], view };
        cleanup = view.stop;
        show(view.element);
      }
      await shown!.view.select(episode);
      return;
    }
    leave();
    if (parts.length === 0) {
      show(await runsPage(me));
    } else {
      show(h("p", null, "There's no such page. ", link("/", "See all runs.")));
    }
  } catch (err) {
    leave();
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

