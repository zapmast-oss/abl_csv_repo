const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/earld/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const ms = fs.readFileSync('csv/out/story/newsroom/issues/1981-08-30_461bc955c6db88b4-observer-conversation/ccb5293571cd46b0b895/observer/the-forecast-meets-the-facts.md', 'utf8');
const copy = ms.split('\n\n').slice(2).join('\n\n').split('\n\n## Reporting record', 1)[0];
const plain = text => text.replace(/\*/g, '').replace(/\s+/g, ' ').trim();
const paragraphs = copy.split(/\n\n+/).filter(block => !block.startsWith('## ')).map(plain);
const headings = [...copy.matchAll(/^## (.+)$/gm)].map(match => match[1]);
const index = pathToFileURL(path.resolve('output/eab/dist/index.html')).href;

(async () => {
    const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
                                            headless: true, chromiumSandbox: true });
    try {
        for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
            const ctx = await browser.newContext({ viewport: { width, height } });
            await ctx.route(/^https?:/, route => route.abort());
            const page = await ctx.newPage();
            await page.goto(index);
            await page.getByRole('link', { name: 'The Forecast Meets the Facts', exact: true }).click();
            assert.equal(await page.locator('h1').innerText(), 'The Forecast Meets the Facts');
            assert.match(await page.locator('.page-title').innerText(), /The Observer Conversation · Commentary/i);
            assert.deepEqual(await page.locator('article.reading h2').allTextContents(), headings);
            assert.deepEqual((await page.locator('article.reading > p').allTextContents()).slice(0, paragraphs.length).map(plain), paragraphs);
            assert.equal(await page.locator('a[href*="this-is-the-test.html"], a[href*="one-game-is-the-new-margin"]').count(), 0);
            assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
            await page.getByRole('link', { name: 'The Observer Conversation archive', exact: true }).click();
            assert.match(await page.locator('main').innerText(), /August 31, 1981[\s\S]*The Forecast Meets the Facts[\s\S]*August 20, 1981[\s\S]*The Pennant Races Come Home/);
            await page.getByRole('link', { name: 'Read the conversation', exact: true }).last().click();
            assert.match(await page.locator('article.reading').innerText(), /editorial reconstruction/);
            await ctx.close();
            console.log(name + ': exact copy, all eight headings, responsive width, no premature companion link, and archive passed');
        }
    } finally {
        await browser.close();
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
