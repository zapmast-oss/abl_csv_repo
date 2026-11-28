const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/earld/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const preview = path.resolve('output/eab/august-31-complete-edition-review');
const dist = path.join(preview, 'dist');
const coverage = JSON.parse(fs.readFileSync(path.join(preview, 'coverage.json'), 'utf8'));
const url = file => pathToFileURL(path.join(dist, file)).href;
const draft = slug => JSON.parse(fs.readFileSync(path.join('tools/newsroom/reviews/1981-08-31', slug + '.json'), 'utf8'));
const normalize = text => text.replace(/\s+/g, ' ').trim();

async function noOverflow(page) {
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), page.url());
}

async function linkTo(list, title, target) {
    const a = list.getByRole('link', {name: title, exact: true});
    assert.equal(await a.count(), 1, title);
    assert.equal(await a.evaluate(node => node.href), target);
    return a;
}

async function checkCopy(page, item) {
    const original = draft(item.slug);
    assert.equal(await page.locator('h1').innerText(), original.title);
    assert.ok((await page.locator('.page-title').innerText()).includes(original.byline));
    const body = page.locator('article.reading');
    const paragraphs = original.sections.flatMap(([, lines]) => lines).map(normalize);
    const actual = (await body.locator(':scope > p').allTextContents()).slice(1, paragraphs.length + 1).map(normalize);
    assert.deepEqual(actual, paragraphs, item.slug);
    assert.deepEqual(await body.locator('h2').allTextContents(), original.sections.map(([label]) => label).filter(Boolean));
    const pdf = body.getByRole('link', {name: 'View PDF', exact: true});
    assert.ok(item.pdf && fs.readFileSync(path.join(dist, item.pdf)).subarray(0, 5).toString() === '%PDF-');
    assert.equal(await pdf.evaluate(node => node.href), url(item.pdf));
    await noOverflow(page);
}

(async () => {
    const browser = await chromium.launch({executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
                                           headless: true, chromiumSandbox: true});
    try {
        for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
            const ctx = await browser.newContext({viewport: {width, height}, isMobile: width < 600, hasTouch: width < 600});
            await ctx.route(/^https?:/, route => route.abort());
            const news = await ctx.newPage();
            const issue = await ctx.newPage();
            const beat = await ctx.newPage();
            const errors = [];
            for (const p of [news, issue, beat]) p.on('pageerror', error => errors.push(error.message));
            await news.goto(url(coverage.newsroom));
            await issue.goto(url(coverage.current_issue));
            assert.match(await issue.locator('h1').innerText(), /August 31/);
            assert.doesNotMatch(await issue.locator('main').innerText(), /no published articles/i);
            const clubLinks = news.locator('#franchise-beats');
            const allStories = news.locator('#articles');
            assert.equal(await clubLinks.getByRole('link').count(), 24);
            for (const item of coverage.clubs) {
                const book = coverage.articles.find(a => a.slug === item.slug);
                const clubLink = clubLinks.locator('a').filter({hasText: item.club + ' beat'});
                assert.equal(await clubLink.count(), 1, item.club);
                assert.equal(await clubLink.evaluate(node => node.href), url(item.club_path) + '#beat');
                const titleUrl = url(item.article_path);
                await linkTo(allStories, item.title, titleUrl);
                await linkTo(issue.locator('main'), item.title, titleUrl);
                await beat.goto(await clubLink.evaluate(node => node.href));
                assert.equal(await beat.locator('h1').innerText(), item.club);
                await noOverflow(beat);
                await (await linkTo(beat.locator('#beat'), item.title, titleUrl)).click();
                assert.equal(beat.url(), titleUrl);
                await checkCopy(beat, book);
            }
            for (const item of coverage.articles.filter(a => a.kind !== 'beat')) {
                const titleUrl = url(item.route);
                await linkTo(allStories, item.title, titleUrl);
                await linkTo(issue.locator('main'), item.title, titleUrl);
                await beat.goto(titleUrl);
                await checkCopy(beat, item);
            }
            await noOverflow(news);
            await noOverflow(issue);
            await news.locator('#observer-conversation').getByRole('link', {name: 'The Forecast Meets the Facts', exact: true}).click();
            assert.equal(await news.locator('h1').innerText(), 'The Forecast Meets the Facts');
            assert.equal(await news.locator('article.reading h2').count(), 8);
            await news.getByRole('link', {name: 'The Observer Conversation archive', exact: true}).click();
            assert.match(await news.locator('main').innerText(), /The Forecast Meets the Facts[\s\S]*The Pennant Races Come Home/);
            await news.getByRole('link', {name: 'Read the conversation', exact: true}).last().click();
            assert.match(await news.locator('article.reading').innerText(), /editorial reconstruction/);
            assert.deepEqual(errors, []);
            await ctx.close();
            console.log(name + ': all 24 franchise page and newsroom links, 26 exact article texts and PDF routes, and both conversation episodes verified');
        }
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
