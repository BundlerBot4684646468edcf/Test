# AMStudio: Google Apps Script test page

A single page served from `script.google.com`: a free project cost calculator, an explanation, an FAQ and a link to AMStudio.

## Deploy (about 5 minutes)

1. **Use a separate Google account**, not the one that holds AMStudio's email or Drive. If Google cracks down on Apps Script spam, it may suspend the owning account.
2. Go to <https://script.google.com> → **New project**.
3. Paste `Code.gs` into the default `Code.gs` file.
4. **File → New → HTML**, name it `Index` (no extension), and paste in `Index.html`.
5. Edit `CONFIG` at the top of `Code.gs`: domain, prices, texts and FAQ.
6. **Deploy → New deployment → Select type: Web app**
   - Execute as: **Me**
   - Who has access: **Anyone**
7. Copy the `https://script.google.com/macros/s/.../exec` URL.

## Get it indexed

Google has to discover the URL before it can rank it:
- Link to it from your own site, from social profiles and from anything already indexed.
- Search `site:script.google.com "AMStudio"` after a few days to check whether it's in the index.

## Notes

- After editing, use **Deploy → Manage deployments → Edit → New version**, or the public URL keeps serving the old code.
- Consumer (non-Workspace) accounts show a Google banner saying the app was made by a user, not by Google. This can't be removed.
- Google can remove the page from search at any time, so keep the main version on AMStudio's own domain.
