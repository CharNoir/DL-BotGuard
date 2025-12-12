import asyncio
import random
import time
from playwright.async_api import async_playwright

WIKI_RANDOM = "https://en.wikipedia.org/wiki/Special:Random"

FAST_SCROLL_RANGE = (300, 1200)
FAST_SCROLL_DELAY = (0.05, 0.25)

GIBBERISH_WORDS = [
    "test", "hello", "bot", "user", "login", "guest", "admin", "root",
    "abc", "abcd", "abc123", "123", "1234", "12345", "0000", "1111",
    "pass", "password", "passwd", "pwd", "login123", "user1", "guest1",

    "qwe", "qwer", "qwerty", "asdf", "zxcv", "aaa", "bbb", "ccc", "xyz",
    "foo", "bar", "foobar", "baz", "lol", "omg", "wtf", "spam", "ping",
    
    "admin123", "test123", "pw123", "letmein", "welcome", "default",
    "123qwe", "qwe123", "pass123", "myname", "useruser",
    "superuser", "root123", "qazwsx", "zaq12wsx", "1q2w3e",

    "select", "update", "info", "token", "email", "login", "session",
    "debug", "check", "verify", "captcha", "submit", "formdata",
    "search", "payload", "string", "flag", "value", "hidden",

    "apple", "banana", "cat", "dog", "mouse", "car", "train", "table",
    "sky", "blue", "rain", "sun", "moon", "night", "light",
    "water", "stone", "road", "music", "house", "page", "data",

    "xD3f", "nmk2", "lp9d", "tr55", "vvv1", "opq9", "xxzz", "bb12",
    "okok", "ok123", "nope", "why", "gggg", "testtest", "idk",
    "hmmm", "random", "zzz", "zzzzzz", "maybe", "tryagain",

    "thisisatest", "randomstring", "mypassword", "justchecking",
    "icanhaz", "someinput", "loremipsum", "hello123", "searching",
    "enteringtext", "botactivity", "generatedbybot",

    "john", "mike", "alex", "anna", "kate", "david", "tom", "emma",
    "jake", "lucy", "mark", "eva", "leo", "mia", "sam", "joel",

    "test@test.com", "user@example.com", "bot@bot.com",
    "random@mail.com", "hello@world.com",
]

def random_gibberish():
    """Return a random small text fragment."""
    if random.random() < 0.5:
        return random.choice(GIBBERISH_WORDS)
    # Or random letters & digits
    length = random.randint(3, 10)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))

async def type_random_gibberish(page):
    """
    Randomly types a short 'bot-like' word.
    Sometimes targets a search bar, sometimes the body.
    """
    word = random_gibberish()

    async def safe_type(selector, text):
        try:
            el = await page.query_selector(selector)
            if not el:
                return False

            await el.click()

            await el.fill("")  
            await el.type(text, delay=random.randint(10, 60))
            
            if random.random() < 0.3:
                el2 = await page.query_selector(selector)
                if el2:
                    await el2.press("Enter")
            return True

        except Exception:
            return False

    if random.random() < 0.4:
        ok = await safe_type("input[name='search']", word)
        if ok:
            return True
        
    inputs = await page.query_selector_all("input[type='text'], input:not([type])")
    if inputs and random.random() < 0.2:
        selector = "input[type='text'], input:not([type])"
        ok = await safe_type(selector, word)
        if ok:
            return True

    try:
        await page.keyboard.type(word, delay=random.randint(5, 50))
        if random.random() < 0.2:
            await page.keyboard.press("Enter")
        if random.random() < 0.1:
            await page.keyboard.press("Tab")
    except:
        pass

    return True

async def fast_scroll(page):
    """Perform several fast downward scrolls."""
    for _ in range(random.randint(5, 20)):
        px = random.randint(*FAST_SCROLL_RANGE)
        await page.mouse.wheel(0, px)
        await asyncio.sleep(random.uniform(*FAST_SCROLL_DELAY))

async def click_random_link(page):
    """
    Click a random /wiki/ link on the page.
    Returns True if a click was performed.
    """
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
    """
    Main autonomous bot loop:
      - open random Wikipedia pages
      - scroll
      - randomly click / type
    """
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

            await fast_scroll(page)

            action = random.choice(["scroll", "click", "type"])

            if action == "click":
                clicked = await click_random_link(page)
                if not clicked:
                    continue

            elif action == "type":
                await type_random_gibberish(page)

            await random_wait()

# Start bot
asyncio.run(wikipedia_bot())