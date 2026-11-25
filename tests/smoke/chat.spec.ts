import { expect, test } from "@playwright/test";

test("a learner sends a message and sees Tutor's reply", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("textbox", { name: "Message Tutor" }).fill("How can I visualize one half?");

  const reply = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/chat" && response.request().method() === "POST"
  );
  await page.getByRole("button", { name: "Send message" }).click();

  expect((await reply).ok()).toBe(true);
  await expect(page.getByText("Try drawing a fraction bar.", { exact: true })).toBeVisible();
});
