# Third-party notices

## assistant-ui

The Thread, Markdown renderer, tooltip button, Button, Tooltip, and clipboard hook
in `src/components/` and `src/hooks/` are adapted from the assistant-ui minimal
template at commit `d56b46509d2a0e3f97494dd3b7b1e1d7fdd4f37f`:

https://github.com/assistant-ui/assistant-ui/tree/d56b46509d2a0e3f97494dd3b7b1e1d7fdd4f37f/templates/minimal

The integration keeps the stock thread layout, Markdown/code copy, message actions,
editing, branch navigation, and scroll behavior. It removes attachment, tool,
dictation, and suggestion UI to match the current text-only application.
Message action bars stay mounted for keyboard and touch access instead of
requiring mouse hover.
Request errors use the app's dismissible banner, so the duplicate per-message
error rendering is omitted.

MIT License

Copyright (c) 2026 AgentbaseAI Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
