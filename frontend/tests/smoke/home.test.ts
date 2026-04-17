import { expect, test } from "@playwright/test";

const MISSION_CONFIGURATION = /mission configuration/i;
const FIND_COMPETITIONS = /find competitions/i;
const BEST_RESULTS = /best results/i;
const VIEW_ALL = /^view all$/i;
const COMPETITIONS_SEARCH = "/api/competitions/search";

test.describe("agents-k.com smoke", () => {
  test("home page renders mission configuration without console errors", async ({
    page,
  }) => {
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(msg.text());
      }
    });

    const response = await page.goto("/");
    expect(response?.status()).toBeLessThan(400);

    await expect(
      page.getByRole("heading", { name: MISSION_CONFIGURATION })
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: FIND_COMPETITIONS })
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: BEST_RESULTS })
    ).toBeVisible();

    expect(consoleErrors).toEqual([]);
  });

  test("competitions search returns competitions through the UI", async ({
    page,
  }) => {
    await page.goto("/");
    const searchResponsePromise = page.waitForResponse((res) =>
      res.url().includes(COMPETITIONS_SEARCH)
    );
    await page.getByRole("button", { name: FIND_COMPETITIONS }).click();
    const searchResponse = await searchResponsePromise;

    expect(searchResponse.status()).toBe(200);
    const body = await searchResponse.json();
    expect(Array.isArray(body.competitions)).toBe(true);
    expect(body.competitions.length).toBeGreaterThan(0);
  });

  test("View All best-results button is hidden when no overflow", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: BEST_RESULTS })
    ).toBeVisible();
    await expect(page.getByRole("button", { name: VIEW_ALL })).toHaveCount(0);
  });
});
