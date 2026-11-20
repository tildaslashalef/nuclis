---
tags: [prompts]
---
# TODO List App 



Design and build a complete, runnable TODO list web application using Bun (`bun run`).

### 1. Architecture & Runtime Requirements
- **Zero external dependencies**: Use only Bun's built-in APIs (`Bun.serve()`, `bun:sqlite`) and standard Web APIs. Do not use Express, Hono, React, or external CSS build tools.
- **Database Persistence**: Use an embedded SQLite database (`./todos.db`) via `bun:sqlite` with WAL mode enabled (`PRAGMA journal_mode = WAL;`). Automatically initialize the database schema on startup if it does not exist.
- **Full-Stack Delivery**: Serve both the REST API (`/api/todos` supporting GET, POST, PATCH, DELETE) and the frontend HTML, CSS, and client-side JavaScript/TypeScript from a single `Bun.serve()` server on port `3000`.
- **Code Quality**: Keep the backend and frontend cleanly organized with clear separation of concerns, no placeholder comments, and no truncated logic.

### 2. Functional Features
- **Task Schema**: `id`, `title`, `description` (optional), `priority` (`low`, `medium`, `high`), `due_date` (optional), `completed` (boolean), and `created_at`.
- **CRUD Operations**:
  - Add tasks with a priority level and optional due date.
  - Toggle completion status with immediate optimistic UI updates.
  - Edit task titles and priorities inline or via a clean modal.
  - Delete individual tasks and provide a "Clear Completed" bulk action.
- **Filtering & Search**: Instant client-side filtering by status (`All`, `Active`, `Completed`), priority filter pills, and real-time keyword search.
- **Keyboard Shortcuts**: Pressing `/` focuses the new task input; `Enter` submits; `Escape` closes active modals or clears inputs.

### 3. Visual Design & UI Polish
Build a cohesive, modern "Linear / Apple Reminders" inspired interface with a dark-mode-first default and a working light/dark theme toggle:
- **Color System**: Define CSS custom properties (`--bg-primary`, `--surface`, `--border`, `--text-primary`, `--text-secondary`, `--accent`). Default to a deep slate dark theme (`#0B0F19` background, `#111827` elevated card surfaces with subtle `1px solid rgba(255,255,255,0.08)` borders, and an indigo accent `#6366F1`).
- **Layout & Typography**: System sans-serif font stack (`-apple-system, BlinkMacSystemFont, "Inter", sans-serif`), `1.5` line-height, generous container padding (`24px`–`32px`), and a centered `760px` max-width container.
- **UI Details**:
  - Subtle backdrop blur (`backdrop-filter: blur(12px)`) on sticky headers and modals.
  - Priority badges styled as soft tinted pills (crimson tint for High, amber tint for Medium, emerald tint for Low).
  - Smooth CSS micro-interactions: custom animated SVG checkboxes, subtle hover elevation on task cards, smooth transitions when adding or removing tasks, and non-intrusive toast notifications for errors.
  - A clean inline SVG empty-state illustration when no tasks match the current view.

### 4. Definition of Done
- The application passes Bun's build/syntax verification without errors and starts cleanly so `./todos.db` is created and ready on `http://localhost:3000`.

Before creating any files, present your proposed folder structure and architectural approach so we can discuss and align on the design first.
