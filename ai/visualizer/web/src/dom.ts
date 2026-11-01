// A tiny element builder, so pages are built from data without innerHTML.

type Child = Node | string | number | null | undefined | false;
type Props = Record<string, unknown> & { class?: string };

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Props | null = null,
  ...children: (Child | Child[])[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = String(value);
    else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value as EventListener);
    } else if (key in el && typeof value !== "string") (el as any)[key] = value;
    else el.setAttribute(key, String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

// Line icons in the Hattracker's style: a 16-unit box, drawn in the text
// color with round ends. Shapes marked solid are filled instead.
const ICONS: Record<string, [string, "solid"?][]> = {
  back: [["M10 3.5 5.5 8l4.5 4.5"]],
  play: [["M5 3.2v9.6a.5.5 0 0 0 .76.43l7.6-4.8a.5.5 0 0 0 0-.86l-7.6-4.8A.5.5 0 0 0 5 3.2z", "solid"]],
  pause: [["M4.5 3h2.4v10H4.5zM9.1 3h2.4v10H9.1z", "solid"]],
  previous: [["M3.5 3.5v9"], ["M12.5 3.6v8.8a.4.4 0 0 1-.62.33L5.6 8.33a.4.4 0 0 1 0-.66l6.28-4.4a.4.4 0 0 1 .62.33z", "solid"]],
  next: [["M12.5 3.5v9"], ["M3.5 3.6v8.8a.4.4 0 0 0 .62.33l6.28-4.4a.4.4 0 0 0 0-.66L4.12 3.27a.4.4 0 0 0-.62.33z", "solid"]],
  plus: [["M8 3v10M3 8h10"]],
  close: [["M4 4l8 8M12 4l-8 8"]],
  signout: [["M6.5 2.5h-3a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h3"], ["M10 5l3 3-3 3M13 8H6"]],
};

export type IconName = keyof typeof ICONS;

/** An icon to put beside a label, or alone in a button with an aria-label. */
export function icon(name: IconName, size = 16): SVGSVGElement {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  svg.classList.add("icon");
  for (const [d, solid] of ICONS[name]) {
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", d);
    if (solid) {
      path.setAttribute("fill", "currentColor");
    } else {
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "currentColor");
      path.setAttribute("stroke-width", "1.5");
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("stroke-linejoin", "round");
    }
    svg.append(path);
  }
  return svg;
}

/** A link that changes page without reloading. */
export function link(href: string, ...children: Child[]): HTMLAnchorElement {
  return h("a", {
    href,
    onclick: (e: MouseEvent) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      e.preventDefault();
      navigate(href);
    },
  }, ...children);
}

let router: () => void = () => {};

export function setRouter(fn: () => void): void {
  router = fn;
  window.addEventListener("popstate", fn);
}

export function navigate(href: string): void {
  history.pushState(null, "", href);
  router();
}

export function when(iso: string | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

export function duration(start?: string, end?: string): string {
  if (!start || !end) return "";
  const s = Math.round((new Date(end).getTime() - new Date(start).getTime()) / 1000);
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}
