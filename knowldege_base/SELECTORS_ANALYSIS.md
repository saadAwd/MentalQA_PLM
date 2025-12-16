# CSS Selectors Analysis - Findings

## Article List Page Structure

Based on analysis of `article_list_page_11.html`:

### Article Links
- **Selector**: `div.articles-list a[href*='/articals/']`
- **Pattern**: Links point to `/articals/{id}` (note: "articals" not "articles" - appears to be a typo in the website)
- **Structure**: 
  ```html
  <div class="articles-list">
    <div>
      <a href="https://ncmh.org.sa/articals/194">
        <div class="col article-card-col">
          <div class="article-card">
            <p class="article-title">...</p>
            <p class="article-date">
              <span class="date">الثلاثاء 5 مارس 2024</span>
            </p>
          </div>
        </div>
      </a>
    </div>
  </div>
  ```

### Article Card Elements (on list page)
- **Title**: `p.article-title` (contains title text + "اقرأ المزيد" link)
- **Date**: `p.article-date span.date` (Arabic date format)

## Article Detail Page

**Note**: The exploration script fetched what appears to be a news list page (`/news`) rather than an actual article detail page (`/articals/{id}`). 

### Recommended Selectors (to be verified):
- **Title**: `h1` (most common pattern)
- **Body**: Try multiple selectors:
  - `div.article-content`
  - `div.article-body`
  - `article`
  - `main .container`
- **Date**: `span.date, .article-date, time` (if available)

## Next Steps

1. **Test Article Scraper**: Run `scraper_articles_poc.py` to scrape page 11 articles
2. **Verify Detail Page Structure**: Once an article is scraped, check the raw HTML file to verify:
   - Title selector works correctly
   - Body content selector extracts the full article text
   - Date selector (if needed)
3. **Update Config**: Adjust selectors in `config.py` based on actual article detail page structure

## Important Notes

- URL pattern uses `/articals/` (typo) not `/articles/`
- Website is RTL (right-to-left) Arabic content
- Dates are in Arabic format
- Article titles may contain embedded "اقرأ المزيد" links that should be cleaned

