# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.
- File preview/download: `GET /files/{file_id}/preview` streams file content from Supabase Storage with correct Content-Type and Content-Disposition: inline headers. Uses `FileService.get_file()` for metadata + `FileService.download_file()` for bytes. Falls back to `mimetypes.guess_type()` based on extension for Content-Type. The route is protected by org membership (via `require_org_membership` on the file router). `FileService.download_file()` wraps Supabase's sync `.download()` in `asyncio.to_thread` for async safety.
