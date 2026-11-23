/* Run browser checks against the local November 23 opening-day build and/or the replay. */
const assert = require('node:assert/strict');
const { mkdirSync } = require('node:fs');
const { join, resolve } = require('node:path');
const { pathToFileURL } = require('node:url');

const [directory, screenshots, modulePath = 'playwright', executablePath] = process.argv.slice(2);
if (!directory || !screenshots) throw new Error('Pass the public folder and the screenshot folder.');
const { chromium } = require(modulePath);
const index = pathToFileURL(resolve(directory, 'index.html')).href;

async function expectText(locator, pattern) {
  assert.match(await locator.innerText(), pattern);
}

async function noOverflow(page) {
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, page.url());
}

async function visitNav(page, label, title) {
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: label, exact: true }).click();
  await expectText(page.locator('h1'), title);
  await noOverflow(page);
}

async function audit(browser, name, viewport) {
  const context = await browser.newContext({ viewport, reducedMotion: 'reduce' });
  await context.route(/^https?:/, route => route.abort());
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(index + '#editions');
  await expectText(page.locator('h1'), /Publication timeline/);
  await page.getByRole('link', { name: 'Open the current edition' }).click();
  const today = page.url();
  await page.goto(index, { waitUntil: 'commit' });
  await page.waitForURL(today);
  await expectText(page.locator('.hero h1'), /first place/i);
  await expectText(page.locator('.edition-bar'), /August 20, 1981.*Finals through August 19/s);
  await expectText(page.locator('.edition-credit time'), /Updated .* UTC/);
  assert.ok(await page.locator('.hero-art img').evaluate(img => img.complete && img.naturalWidth > 0));
  await noOverflow(page);
  await page.screenshot({ path: join(screenshots, `${name}-today.png`) });

  await visitNav(page, 'Newsroom', /ABL Newsroom/);
  const beatLinks = await page.locator('a.team-tile').evaluateAll(links => links.map(a => [a.href, a.querySelector('b').textContent]));
  assert.equal(new Set(beatLinks.map(([href]) => href)).size, 24);
  if (name === 'desktop') {
    for (const [href, beat] of beatLinks) {
      await page.goto(href);
      assert.equal(await page.locator('h1').innerText(), beat.replace(/ beat$/, ''));
      assert.equal(await page.getByRole('heading', { name: beat, exact: true }).count(), 1);
      assert.equal(new URL(href).hash, '#beat');
      await noOverflow(page);
    }
    await visitNav(page, 'Newsroom', /ABL Newsroom/);
  }
  await page.getByRole('link', { name: 'Open current issue' }).click();
  await expectText(page.locator('main'), /Games through August 19, 1981/);
  const hasStories = await page.getByRole('link', { name: 'Permanent issue link' }).count();
  if (hasStories) {
    await page.getByRole('link', { name: 'Permanent issue link' }).click();
    await page.getByRole('link', { name: /couldn.t shake Phoenix/ }).click();
    await expectText(page.locator('main'), /Ernie Bewell.*August 20, 1981/s);
    await expectText(page.locator('.reading'), /Games through August 19, 1981/);
    assert.match(await page.getByRole('link', { name: 'View PDF' }).getAttribute('href'), /\.pdf$/);
    await page.screenshot({ path: join(screenshots, `${name}-article.png`) });
    await page.goto(today);
    await visitNav(page, 'Newsroom', /ABL Newsroom/);
    await page.getByRole('searchbox', { name: 'Search article archive' }).fill('Ernie Phoenix');
    assert.equal(await page.locator('#articles [data-item]:visible').count(), 1);
    const issueLinks = await page.locator('main .list a').evaluateAll(links => links.map(link => link.href));
    assert.ok(new Set(issueLinks).size > 1, 'Earlier issue must have a distinct URL');
    await page.goto(issueLinks.at(-1));
    await expectText(page.locator('main'), /Games through August 19, 1981/);
    assert.equal(await page.locator('main .tile').count(), 1);
  } else {
    await expectText(page.locator('main'), /This date has no published articles/);
  }

  await visitNav(page, 'Teams', /The 24 clubs/);
  await page.getByRole('link', { name: /Phoenix Firebirds/ }).click();
  await expectText(page.locator('h1'), /Phoenix Firebirds/);
  await expectText(page.locator('main'), /Recent results.*Upcoming schedule/s);
  await page.getByRole('link', { name: 'Club statistics' }).click();
  await expectText(page.locator('h1'), /Phoenix Firebirds statistics/);
  await expectText(page.locator('main'), /Team batting.*Team pitching/s);
  await visitNav(page, 'Standings', /Standings & races/);
  assert.equal(await page.getByRole('table', { name: 'Division standings' }).count(), 6);
  assert.equal(await page.getByText('Western Division', { exact: true }).count(), 2);
  await expectText(page.locator('main'), /three division winners and one wild card/);
  await page.screenshot({ path: join(screenshots, `${name}-standings.png`) });

  await visitNav(page, 'Games', /Game Center/);
  await page.getByRole('searchbox', { name: 'Find a game date' }).fill('1981-08-19');
  assert.equal(await page.locator('#game-dates li:visible').count(), 1);
  await page.locator('#game-dates li:visible a').click();
  await expectText(page.locator('h1'), /August 19/);
  const box = page.locator('#g-1446');
  await expectText(box, /Final · 2–18/);
  await box.locator('summary').click();
  assert.ok(await box.getByRole('table').first().isVisible());
  await page.getByRole('link', { name: 'Next date' }).click();
  const unfinished = page.locator('#g-1113');
  await expectText(unfinished, /Unfinished game/);
  assert.equal(await unfinished.locator('summary').count(), 0);
  await expectText(page.locator('#g-1458'), /Scheduled/);

  await visitNav(page, 'Players', /Players & statistics/);
  await page.getByRole('link', { name: 'All individual batting' }).click();
  await page.getByRole('searchbox', { name: 'Find a player or club' }).fill('Luke Margalit');
  assert.equal(await page.locator('#player-table tbody tr:visible').count(), 1);
  await expectText(page.locator('#player-table tbody tr:visible'), /Luke Margalit/);
  await noOverflow(page);

  await visitNav(page, 'History', /History & archives/);
  await page.locator('.tile').filter({ hasText: '1980' }).getByRole('link', { name: 'Open season' }).click();
  await expectText(page.locator('h1'), /1980 season almanac/);
  await expectText(page.locator('.page-title'), /Detroit/);
  await page.getByRole('link', { name: 'Browse 1980 player records' }).click();
  await expectText(page.locator('h1'), /1980 player records/);
  assert.ok(await page.locator('#historical-pitching tbody tr').count() > 300);
  await noOverflow(page);

  await page.goto(today);
  await visitNav(page, 'Fan Culture', /Fan Culture/);
  await page.getByRole('searchbox', { name: 'Find a club or ballpark' }).fill('Phoenix');
  assert.equal(await page.locator('#attendance tbody tr:visible').count(), 1);
  await visitNav(page, 'Sim Baseball Vision', /Sim Baseball Vision/);
  const links = await page.locator('main a[href^="https:"]').evaluateAll(anchors => anchors.map(a => a.href));
  assert.ok(links.some(link => link.includes('VzDa04It9lE')));
  assert.ok(links.some(link => link.includes('showthread.php?t=366841')));
  assert.ok(links.some(link => link.includes('/channel/UCd1wShl1bahaTldUtqVZVwg')));
  await page.getByRole('link', { name: 'Browse the game calendar' }).click();
  await expectText(page.locator('h1'), /Game Center/);
  await visitNav(page, 'About', /See it for yourself/);
  await expectText(page.locator('main'), /fictional professional baseball league/);
  await page.getByRole('link', { name: 'Visit the publication timeline' }).click();
  await expectText(page.locator('h1'), /Publication timeline/);
  if (hasStories) {
    await page.getByRole('link', { name: /Finals through August 18/ }).click();
    await expectText(page.locator('.edition-bar'), /Finals through August 18/);
  }
  assert.deepEqual(errors, []);
  await context.close();
  console.log(`${name}: navigation, issue and archive, 1981 data, historical stats, searches and layout passed`);
}

(async () => {
  mkdirSync(screenshots, { recursive: true });
  const browser = await chromium.launch({ executablePath, headless: true, chromiumSandbox: true });
  try {
    await audit(browser, 'desktop', { width: 1440, height: 1000 });
    await audit(browser, 'tablet', { width: 820, height: 1180 });
    await audit(browser, 'mobile', { width: 390, height: 844 });
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
