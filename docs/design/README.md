UI design - AI-Generated Video Forgery Detection (Frontend / UX)
Figma Link: https://www.figma.com/design/JQiUO9m2OELckMOOYwX6m8/AI-Video-Forgery-Detection---UI-Mockups?node-id=0-1&t=trSBEQGO26Cq0hD2-1

Files

wireframes.png - low-fidelity wireframes of the 5 mobile screens, with the API call that feeds each one

mockups_overview_mobile.png - overview of the mobile mockups

mockups/ - full-resolution PNGs (mobile 360 x 780 @2x, desktop 1440 x 900 @2x)

m1-upload, m1b-upload-file-selected, m2-processing, m3a-result-real, m3b-result-forged, m4-history, m5-errors

d1-upload, d3-result-forged, d4-history

html/ - the HTML/CSS source of every screen (reuse it when building the React components)

Design choices

Font: Inter (falls back to Segoe UI / Arial). Corner radius 12-16 px.

Colours: indigo #4f46e5 (primary actions), green #15803d / #dcfce7 (REAL), red #b91c1c / #fee2e2 (FORGED), slate greys for text and borders.

Result screen: verdict first (largest element), then probability meter with a 50% marker, then frame-by-frame bars, then details.

Responsive: mobile first (320-360 px); desktop uses two columns for the result and a table for history (NFR-003).

Data shown

Result screens use the values from two real test runs of the backend (REAL 0.26, FORGED 0.81). History rows are sample data.

The "steps" list on the processing screen is illustrative; the API only reports queued / processing / completed / failed.

API mapping

Upload: POST /api/v1/videos, then POST /api/v1/videos/{id}/analyze

Status and result: GET /api/v1/jobs/{id} (poll every 1.5 s)

History: GET /api/v1/jobs?limit=20&offset=0

Delete: DELETE /api/v1/videos/{id}