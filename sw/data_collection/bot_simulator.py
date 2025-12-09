import asyncio
import random
import time
from playwright.async_api import async_playwright

WIKI_RANDOM = "https://en.wikipedia.org/wiki/Special:Random"

FAST_SCROLL_RANGE = (300, 1200)
FAST_SCROLL_DELAY = (0.05, 0.25)

GIBBERISH_WORDS = [
    # simple words
    "test", "hello", "bot", "user", "login", "guest", "admin", "root",
    "abc", "abcd", "abc123", "123", "1234", "12345", "0000", "1111",
    "pass", "password", "passwd", "pwd", "login123", "user1", "guest1",

    # random short junk
    "qwe", "qwer", "qwerty", "asdf", "zxcv", "aaa", "bbb", "ccc", "xyz",
    "foo", "bar", "foobar", "baz", "lol", "omg", "wtf", "spam", "ping",
    
    # more “password-like” patterns
    "admin123", "test123", "pw123", "letmein", "welcome", "default",
    "123qwe", "qwe123", "pass123", "myname", "useruser",
    "superuser", "root123", "qazwsx", "zaq12wsx", "1q2w3e",

    # technical/suspicious bot strings
    "select", "update", "info", "token", "email", "login", "session",
    "debug", "check", "verify", "captcha", "submit", "formdata",
    "search", "payload", "string", "flag", "value", "hidden",

    # generic random dictionary-like words
    "apple", "banana", "cat", "dog", "mouse", "car", "train", "table",
    "sky", "blue", "rain", "sun", "moon", "night", "light",
    "water", "stone", "road", "music", "house", "page", "data",

    # bot-like nonsense typing
    "xD3f", "nmk2", "lp9d", "tr55", "vvv1", "opq9", "xxzz", "bb12",
    "okok", "ok123", "nope", "why", "gggg", "testtest", "idk",
    "hmmm", "random", "zzz", "zzzzzz", "maybe", "tryagain",

    # longer junk bots often enter
    "thisisatest", "randomstring", "mypassword", "justchecking",
    "icanhaz", "someinput", "loremipsum", "hello123", "searching",
    "enteringtext", "botactivity", "generatedbybot",

    # fake names
    "john", "mike", "alex", "anna", "kate", "david", "tom", "emma",
    "jake", "lucy", "mark", "eva", "leo", "mia", "sam", "joel",

    # fake emails (bots often type these)
    "test@test.com", "user@example.com", "bot@bot.com",
    "random@mail.com", "hello@world.com",
]

def random_gibberish():
    """Generate a random short word the way bots type."""
    if random.random() < 0.5:
        return random.choice(GIBBERISH_WORDS)
    # Or random letters & digits
    length = random.randint(3, 10)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))

async def type_random_gibberish(page):
    """Bot-like typing activity with full DOM-safety."""
    word = random_gibberish()

    # Helper: safely type into selector
    async def safe_type(selector, text):
        try:
            el = await page.query_selector(selector)
            if not el:
                return False

            # Click fresh element
            await el.click()

            # Fill fresh element
            await el.fill("")  
            await el.type(text, delay=random.randint(10, 60))
            
            # Chance to press Enter
            if random.random() < 0.3:
                el2 = await page.query_selector(selector)
                if el2:
                    await el2.press("Enter")
            return True

        except Exception:
            return False

    # 40% chance: try Wikipedia search bar
    if random.random() < 0.4:
        ok = await safe_type("input[name='search']", word)
        if ok:
            return True

    # 20% chance: try any other input field
    inputs = await page.query_selector_all("input[type='text'], input:not([type])")
    if inputs and random.random() < 0.2:
        selector = "input[type='text'], input:not([type])"
        ok = await safe_type(selector, word)
        if ok:
            return True

    # Else: type directly into the page body
    try:
        await page.keyboard.type(word, delay=random.randint(5, 50))
        if random.random() < 0.2:
            await page.keyboard.press("Enter")
        if random.random() < 0.1:
            await page.keyboard.press("Tab")
    except:
        pass

    return True

    # 20% chance: type into any visible input
    inputs = await page.query_selector_all("input[type='text'], input:not([type])")
    if inputs and random.random() < 0.2:
        inp = random.choice(inputs)
        try:
            await inp.click()
            await inp.type(word, delay=random.randint(5, 40))
            if random.random() < 0.2:
                await inp.press("Enter")
            return True
        except:
            pass

    # Else: type into the page body (no field)
    await page.keyboard.type(word, delay=random.randint(5, 50))

    # Sometimes press Enter or Tab
    if random.random() < 0.2:
        await page.keyboard.press("Enter")
    if random.random() < 0.1:
        await page.keyboard.press("Tab")

    return True

async def fast_scroll(page):
    for _ in range(random.randint(5, 20)):
        px = random.randint(*FAST_SCROLL_RANGE)
        await page.mouse.wheel(0, px)
        await asyncio.sleep(random.uniform(*FAST_SCROLL_DELAY))

async def click_random_link(page):
    links = await page.query_selector_all("a[href^='/wiki/']")
    if not links:
        return False

    link = random.choice(links)
    box = await link.bounding_box()
    if not box:
        return False

    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2

    await page.mouse.move(x, y)
    await page.mouse.click(x, y)
    return True

async def random_wait():
    await asyncio.sleep(random.uniform(0.2, 1.5))

async def wikipedia_bot():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        while True:
            try:
                async with page.expect_navigation():
                    await page.goto(WIKI_RANDOM)
            except:
                pass
            
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
            except:
                pass

            # Phase 1: Scroll
            await fast_scroll(page)

            # Phase 2: Random bot choice
            action = random.choice(["scroll", "click", "type"])

            if action == "click":
                clicked = await click_random_link(page)
                if not clicked:
                    continue

            elif action == "type":
                await type_random_gibberish(page)

            # Phase 3: Small delay
            await random_wait()

asyncio.run(wikipedia_bot())