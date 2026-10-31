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
