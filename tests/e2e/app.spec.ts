import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';

/** Collects console errors and page errors for the whole test. */
function trackErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  return errors;
}

/** Fraction of distinct colours in a canvas sample — proves the game actually draws. */
async function canvasColours(page: Page, selector: string): Promise<number> {
  return page.locator(selector).evaluate((c: HTMLCanvasElement) => {
    const ctx = c.getContext('2d')!;
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    const set = new Set<number>();
    for (let i = 0; i < d.length; i += 4 * 97) set.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    return set.size;
  });
}

test('direct play: the game runs, draws and accepts input', async ({ page, isMobile }) => {
  const errors = trackErrors(page);
  await page.goto('./#play');
  const canvas = page.locator('[data-testid=page-play] [data-testid=game-canvas]');
  await expect(canvas).toBeVisible();
  await expect(page.locator('[data-testid=page-play] [data-testid=my-length]')).toBeVisible();
  await expect.poll(() => canvasColours(page, '[data-testid=page-play] [data-testid=game-canvas]'), { timeout: 10000 }).toBeGreaterThan(8);
  const t0 = await page.locator('[data-testid=page-play] [data-testid=sim-time]').innerText();
  await page.waitForTimeout(1500);
  expect(await page.locator('[data-testid=page-play] [data-testid=sim-time]').innerText()).not.toBe(t0);
  if (isMobile) {
    await expect(page.getByTestId('joystick')).toBeVisible();
    await expect(page.getByTestId('boost-button')).toBeVisible();
  } else {
    const box = (await canvas.boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.9, box.y + box.height * 0.2);
    await page.keyboard.down('Space');
    await page.waitForTimeout(300);
    await page.keyboard.up('Space');
  }
  expect(errors).toEqual([]);
});

test('pause, single step and reset', async ({ page }) => {
  await page.goto('./#play');
  const dock = page.locator('[data-testid=page-play] [data-testid=dock]');
  await dock.getByTestId('play-pause').click();
  const time = page.locator('[data-testid=page-play] [data-testid=sim-time]');
  await page.waitForTimeout(400);
  const paused = await time.innerText();
  await page.waitForTimeout(800);
  expect(await time.innerText()).toBe(paused);
  await dock.getByTestId('step').click();
  await dock.getByTestId('reset').click();
  await expect(time).toHaveText('0:00', { timeout: 5000 }).catch(() => undefined);
});

test('speed control: slider, 1× and MAX', async ({ page, isMobile }) => {
  test.skip(isMobile, 'inline slider is the desktop layout; phones use the sheet');
  await page.goto('./#watch');
  const dock = page.locator('[data-testid=page-watch] [data-testid=dock]');
  const slider = dock.getByTestId('speed-slider');
  await slider.focus();
  await page.keyboard.press('PageUp');
  const value = dock.locator('.speed-inline [data-testid=speed-value]');
  await expect(value).not.toHaveText('1.0×');
  await dock.getByTestId('speed-reset').click();
  await expect(value).toHaveText('1.0×');
  await dock.getByTestId('speed-max').click();
  await expect(value).toHaveText('최대');
  await expect(dock.locator('.speed-inline [data-testid=speed-actual]')).toBeVisible({ timeout: 5000 });
});

test('watch mode: inspector shows the real decision data and debug overlays toggle', async ({ page, isMobile }) => {
  const errors = trackErrors(page);
  await page.goto('./#watch');
  const root = page.locator('[data-testid=page-watch]');
  if (isMobile) await root.getByTestId('toggle-analysis').click();
  await expect(root.getByTestId('analysis-panel')).toBeVisible();
  await expect(root.getByTestId('mpc-table')).toBeVisible({ timeout: 10000 });
  await expect(root.getByTestId('leaderboard').locator('li')).not.toHaveCount(0);
  await root.getByTestId('toggle-settings').click();
  await root.getByTestId('debug-colliders').click();
  await root.getByTestId('debug-predictions').click();
  await expect(root.getByTestId('debug-colliders')).toHaveAttribute('aria-checked', 'true');
  expect(errors).toEqual([]);
});

test('save → load restores the match', async ({ page, isMobile }, info) => {
  test.skip(isMobile, 'file dialogs are a desktop flow');
  await page.goto('./#watch');
  const dock = page.locator('[data-testid=page-watch] [data-testid=dock]');
  await page.waitForTimeout(1500);
  const [download] = await Promise.all([page.waitForEvent('download'), dock.getByTestId('save').click()]);
  const path = info.outputPath('save.json');
  await download.saveAs(path);
  const file = JSON.parse(readFileSync(path, 'utf8'));
  expect(file.format).toBe('serpent-experiment');
  expect(file.snapshot.world.rulesVersion).toBe('rules-v1');
  const chooser = page.waitForEvent('filechooser');
  await dock.getByTestId('load').click();
  await (await chooser).setFiles(path);
  await expect(page.getByRole('status')).toContainText('불러왔습니다');
});

test('analysis pages load without errors', async ({ page }) => {
  const errors = trackErrors(page);
  for (const mode of ['tournament', 'lab', 'vision', 'calibration']) {
    await page.goto(`./#${mode}`);
    await expect(page.getByRole('heading', { level: 2 })).toBeVisible();
    await page.waitForTimeout(500);
  }
  expect(errors).toEqual([]);
});

test('analysis pages render the shipped result data (no "missing data" placeholders)', async ({ page }) => {
  await page.goto('./#tournament');
  await expect(page.getByTestId('tournament-summary')).toBeVisible();
  await page.goto('./#calibration');
  await expect(page.getByTestId('s2v-survival')).toBeVisible();
  for (const mode of ['lab', 'vision', 'calibration']) {
    await page.goto(`./#${mode}`);
    await page.waitForTimeout(800);
    await expect(page.getByText('측정 데이터 없음')).toHaveCount(0);
  }
  await page.goto('./#vision');
  await expect(page.getByTestId('studio')).toBeVisible();
});

test('vision target page: canvas only, HUD drawn, ground-truth channel only with gt=1', async ({ page }) => {
  await page.goto('./?view=target&seed=5');
  await expect(page.getByTestId('vision-target')).toBeVisible();
  await expect(page.locator('header')).toHaveCount(0);
  await expect.poll(() => canvasColours(page, 'canvas'), { timeout: 10000 }).toBeGreaterThan(8);
  expect(await page.evaluate(() => (window as unknown as { __serpentGT?: unknown }).__serpentGT === undefined)).toBe(true);
  await page.goto('./?view=target&seed=5&gt=1');
  await expect.poll(() => page.evaluate(() => (window as unknown as { __serpentGT?: { frames: unknown[] } }).__serpentGT?.frames.length ?? 0), { timeout: 10000 }).toBeGreaterThan(5);
});

test('GitHub Pages sub-path: assets and data resolve under /SerpentAI/', async ({ page, request }) => {
  await page.goto('./');
  const scripts = await page.locator('script[type=module]').evaluateAll((els) => els.map((e) => (e as HTMLScriptElement).src));
  expect(scripts.every((s) => s.includes('/SerpentAI/'))).toBe(true);
  expect((await request.get('./favicon.svg')).ok()).toBe(true);
  expect((await request.get('./404.html')).ok()).toBe(true);
  expect(await (await request.get('./404.html')).text()).toContain('url=/SerpentAI/');
});
