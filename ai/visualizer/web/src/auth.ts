// Sign-in, the same way `wildhat login` does it: Firebase sends an email
// link, the link brings you back here with a one-time code, and Firebase
// trades the code for an ID token (good for an hour) and a refresh token.
// Both are kept in this browser's localStorage.

const IDENTITY_TOOLKIT = "https://identitytoolkit.googleapis.com/v1";
const SECURE_TOKEN = "https://securetoken.googleapis.com/v1/token";
const SAVED = "wildhat.signin";
const EMAIL = "wildhat.email";
// Where to go once signed in.
const NEXT = "wildhat.next";

interface Saved {
  idToken: string;
  refreshToken: string;
  expiresAt: number; // ms since 1970
}

let apiKey = "";

export function configure(key: string): void {
  apiKey = key;
}

function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null; // e.g. blocked site data
  }
}

function load(): Saved | null {
  try {
    const text = storage()?.getItem(SAVED);
    return text ? (JSON.parse(text) as Saved) : null;
  } catch {
    return null;
  }
}

function save(saved: Saved | null): void {
  try {
    if (saved) storage()?.setItem(SAVED, JSON.stringify(saved));
    else storage()?.removeItem(SAVED);
  } catch {
    // Signed in for this page only.
  }
}

async function post(url: string, body: string, type: string): Promise<any> {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": type }, body });
  const data = await response.json();
  if (!response.ok) {
    const code: string = data?.error?.message ?? `HTTP ${response.status}`;
    throw new Error(MESSAGES[code.split(" ")[0]] ?? `Sign-in failed: ${code}`);
  }
  return data;
}

const MESSAGES: Record<string, string> = {
  INVALID_OOB_CODE: "That sign-in link has expired or was already used. Send a new one.",
  EXPIRED_OOB_CODE: "That sign-in link has expired. Send a new one.",
  INVALID_EMAIL: "That email address doesn't look right.",
  UNAUTHORIZED_DOMAIN: "This site isn't allowed to send sign-in links yet. Ask a mentor.",
  TOKEN_EXPIRED: "Your sign-in has expired. Sign in again.",
};

export function signedIn(): boolean {
  return load() !== null;
}

export async function sendLink(email: string): Promise<void> {
  await post(
    `${IDENTITY_TOOLKIT}/accounts:sendOobCode?key=${apiKey}`,
    JSON.stringify({
      requestType: "EMAIL_SIGNIN",
      email,
      // No query string: Firebase's page adds its own (?oobCode=...).
      continueUrl: `${location.origin}/signin`,
      canHandleCodeInApp: true,
    }),
    "application/json",
  );
  try {
    storage()?.setItem(EMAIL, email);
    storage()?.setItem(NEXT, location.pathname);
  } catch {
    // We'll ask for the email again when the link comes back.
  }
}

/** If this page was opened from a sign-in link, finish signing in and return
 * where to go next. */
export async function finishFromLink(): Promise<string | null> {
  const params = new URLSearchParams(location.search);
  const code = params.get("oobCode");
  if (!code) return null;
  let email: string | null = null;
  try {
    email = storage()?.getItem(EMAIL) ?? null;
  } catch {
    email = null;
  }
  email ??= window.prompt("Which email address did you sign in with?");
  if (!email) throw new Error("Sign-in needs the email address the link was sent to.");
  const data = await post(
    `${IDENTITY_TOOLKIT}/accounts:signInWithEmailLink?key=${apiKey}`,
    JSON.stringify({ email, oobCode: code }),
    "application/json",
  );
  remember(data.idToken, data.refreshToken, Number(data.expiresIn));
  let next: string | null = null;
  try {
    next = storage()?.getItem(NEXT) ?? null;
  } catch {
    next = null;
  }
  return next && next.startsWith("/") && !next.startsWith("//") && next !== "/signin" ? next : "/";
}

function remember(idToken: string, refreshToken: string, expiresIn: number): void {
  save({ idToken, refreshToken, expiresAt: Date.now() + expiresIn * 1000 });
}

/** A current ID token, refreshed if it's about to expire, or null if signed
 * out. */
export async function idToken(): Promise<string | null> {
  const saved = load();
  if (!saved) return null;
  if (saved.expiresAt - Date.now() > 5 * 60 * 1000) return saved.idToken;
  try {
    const data = await post(
      `${SECURE_TOKEN}?key=${apiKey}`,
      new URLSearchParams({ grant_type: "refresh_token", refresh_token: saved.refreshToken }).toString(),
      "application/x-www-form-urlencoded",
    );
    remember(data.id_token, data.refresh_token, Number(data.expires_in));
    return data.id_token;
  } catch {
    signOut();
    return null;
  }
}

export function signOut(): void {
  save(null);
}
