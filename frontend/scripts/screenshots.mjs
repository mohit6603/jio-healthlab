/**
 * Capture README screenshots from the running stack.
 *
 *   docker compose up -d
 *   cd frontend && npm run screenshots
 *
 * Uses the locally installed Chrome rather than downloading a browser, so this
 * works on a machine that already has one. Signs in as a real user, because
 * every interesting page is behind auth.
 */
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE_URL ?? "http://localhost";
const EMAIL = process.env.DEMO_EMAIL ?? "admin@jiohealthlab.example.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "ChangeMe!Admin123";
const OUT = new URL("../../docs/screenshots/", import.meta.url).pathname;

// Chrome's own binary; Playwright's bundled Chromium is not required.
const CHROME =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

mkdirSync(OUT, { recursive: true });

const shot = async (page, name) => {
  await page.screenshot({ path: `${OUT}${name}.png`, fullPage: false });
  console.log(`  ${name}.png`);
};

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2 // retina, so the images stay sharp in the README
});

console.log("Capturing:");

await page.goto(`${BASE}/login`, { waitUntil: "networkidle" });
await shot(page, "login");

await page.fill('input[type="email"]', EMAIL);
await page.fill('input[type="password"]', PASSWORD);
await page.click('button[type="submit"]');
await page.waitForSelector("table", { timeout: 30_000 });
await page.waitForTimeout(1200);
await shot(page, "dashboard");

await page.click('a[href="/assistant"]');
await page.waitForSelector('input#assistant-question', { timeout: 15_000 });
await page.waitForTimeout(600);
await shot(page, "assistant-empty");

// Ask a real question and let it stream, so the screenshot shows a genuine
// grounded answer rather than a mock.
await page.click("text=What does a CBC test measure?");
await page.waitForTimeout(45_000);
try {
  await page.click("text=/\\d+ sources?/", { timeout: 5_000 });
  await page.waitForTimeout(600);
} catch {
  console.log("  (sources toggle not ready; capturing without it expanded)");
}
await shot(page, "assistant-answer");

await page.goto(`${BASE}/analytics`, { waitUntil: "networkidle" });
await page.waitForTimeout(4_000);
await shot(page, "analytics");

await browser.close();
console.log("Done.");
