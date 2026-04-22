# MLAF Testing & Deployment Guide
## How to Make MLAF Available for Testing
### System Author: Neil Shankar Ray
### Provisional Patent Filed — All deployment options preserve IP protection

---

## IMPORTANT: IP Protection Notice

MLAF is under provisional patent. **Do NOT:**
- Push source code to any public repository (GitHub, GitLab, Bitbucket)
- Deploy without authentication to any public URL
- Share the `src/` directory with anyone
- Allow access to `node_modules/` or `package.json` in production

**What is safe to share:**
- The compiled `dist/` folder (minified, bundled JS — not human-readable source)
- Password-protected deployments of the compiled build
- Local demos on your own device

---

## Option 1: Local Demo (Most Secure)

**Best for:** The Kolkata neuro-expert meeting, investor demos, patent review sessions

**How it works:** You bring your laptop. Experts test MLAF right there. No code leaves your machine.

### Setup

```bash
# Step 1: Navigate to the project
cd "/home/ishu/Desktop/gesture grammar"

# Step 2: Install dependencies (if not already done)
npm install

# Step 3: Build the production bundle
npm run build

# Step 4: Serve the production build locally
npm run preview
```

This serves the production build at `http://localhost:4173`.

### At the Meeting

1. Connect your laptop to the projector/display
2. Open Chrome and go to `http://localhost:4173`
3. Grant camera permission when prompted
4. Select an accessibility profile and demonstrate
5. Let experts try gestures on your device

### Pros
- Zero IP exposure — nothing leaves your machine
- Full control over the demo environment
- Works offline (no internet needed at the venue)
- No setup required beyond your laptop + camera

### Cons
- Only works on your device
- Multiple testers must take turns on one machine

---

## Option 2: Private LAN Server (For a Clinic or Lab)

**Best for:** Clinical testing sessions where multiple devices in the same room need access

**How it works:** Your laptop serves MLAF over the local WiFi network. Other devices (tablets, phones) on the same network access it via your laptop's IP address. Nothing goes to the internet.

### Setup

```bash
# Step 1: Build the production bundle
cd "/home/ishu/Desktop/gesture grammar"
npm run build

# Step 2: Find your local IP address
hostname -I
# Example output: 192.168.1.105

# Step 3: Serve on the local network
npx vite preview --host 0.0.0.0 --port 4173
```

Other devices on the same WiFi can now access: `http://192.168.1.105:4173`

### IMPORTANT: Camera Access Requires HTTPS

Browsers block camera access on non-localhost HTTP connections. For LAN testing, you need HTTPS with a self-signed certificate:

```bash
# Step 1: Generate a self-signed SSL certificate (valid for 30 days)
openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -days 30 -nodes \
  -subj '/CN=mlaf-test'

# Step 2: Serve with HTTPS
npx vite preview --host 0.0.0.0 --port 4173 --https --cert cert.pem --key key.pem
```

Other devices access: `https://192.168.1.105:4173`

Testers will see a browser warning ("Your connection is not private") — they click "Advanced" > "Proceed" to continue. This is expected with self-signed certificates.

### At the Clinic

1. Connect your laptop and testing devices to the same WiFi network
2. Run the HTTPS server on your laptop
3. On each testing device (tablet/phone), open Chrome and navigate to `https://<your-ip>:4173`
4. Accept the self-signed certificate warning
5. Grant camera permission
6. Each device can now run MLAF independently

### Pros
- Multiple testers can use different devices simultaneously
- No internet connection required
- Source code stays on your laptop — devices only receive compiled JS
- Clinicians can test on tablets while you observe

### Cons
- All devices must be on the same physical WiFi network
- Self-signed certificate warning (one-time click-through per device)
- Your laptop must stay running as the server

---

## Option 3: Password-Protected Cloud Deployment (For Remote Testers)

**Best for:** Sending MLAF to remote experts, collaborating clinicians in other cities, or review panels who cannot attend in person

**How it works:** You deploy only the compiled `dist/` folder (not source code) to a hosting platform with password protection enabled. Testers receive a URL and password.

### Option 3a: Netlify (Drag & Drop — Simplest)

```bash
# Step 1: Build the production bundle
cd "/home/ishu/Desktop/gesture grammar"
npm run build
```

Then:

1. Go to https://app.netlify.com (create a free account if needed)
2. Drag and drop the `dist/` folder onto the Netlify dashboard
3. Your site deploys instantly with a random URL (e.g., `https://random-name-12345.netlify.app`)
4. Go to **Site Settings** > **Access Control** > **Visitor Access**
5. Select **Password Protection** and set a strong password
6. Share the URL + password with approved testers only

### Option 3b: Vercel (If You Already Have an Account)

You already have a `vercel.json` in the project.

```bash
# Step 1: Build
npm run build

# Step 2: Deploy (Vercel CLI)
npx vercel deploy --prod
```

Then:

1. Go to the Vercel dashboard
2. Navigate to your project settings
3. Enable **Password Protection** (requires Vercel Pro plan, ~$20/month)
4. Set a password
5. Share the URL + password with approved testers

### Option 3c: Cloudflare Pages (Free Password Protection via Workers)

```bash
# Step 1: Build
npm run build
```

Then:

1. Go to https://dash.cloudflare.com > **Pages**
2. Create a new project > Upload assets > Upload the `dist/` folder
3. Add a Cloudflare Worker for basic authentication:

```javascript
// Worker script for basic auth (add via Cloudflare dashboard)
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  const auth = request.headers.get('Authorization');
  const expected = 'Basic ' + btoa('mlaf:YOUR_STRONG_PASSWORD_HERE');

  if (auth !== expected) {
    return new Response('Authentication required', {
      status: 401,
      headers: { 'WWW-Authenticate': 'Basic realm="MLAF Testing"' },
    });
  }

  return fetch(request);
}
```

4. Replace `YOUR_STRONG_PASSWORD_HERE` with a strong password
5. Share credentials with approved testers

### Security Note for All Cloud Options

- The `dist/` folder contains **compiled, minified JavaScript** — not your source code
- However, determined reverse-engineering is theoretically possible (JS is never truly compiled)
- For maximum protection, do NOT deploy to public URLs without password protection
- Change the password after each testing cycle
- Revoke access when testing is complete (delete the deployment)

### Pros
- Remote testers anywhere in the world
- No physical presence required
- Password-gated access
- Only compiled code is exposed (not source)

### Cons
- Compiled JS is technically reverse-engineerable (low risk, but non-zero)
- Requires internet for testers
- Requires a hosting account (free tier is sufficient for all three options)

---

## Option 4: USB Drive Distribution (For Clinics Without Internet)

**Best for:** Rural clinics, government schools, locations with unreliable internet, long-term clinical trials

**How it works:** You build MLAF, copy the compiled output to a USB drive, and deliver it to the clinic. They plug it into any computer with Chrome and a camera.

### Setup

```bash
# Step 1: Build the production bundle
cd "/home/ishu/Desktop/gesture grammar"
npm run build

# Step 2: Copy dist/ folder to USB drive
cp -r dist/ /media/ishu/USB_DRIVE/MLAF/

# Step 3: Create a launcher script on the USB drive
```

Create a file called `start_mlaf.sh` on the USB drive:

```bash
#!/bin/bash
# MLAF Launcher Script
# Requires: Node.js installed on the target machine
# If Node.js is not available, use Python alternative below

echo "Starting MLAF..."
echo "Open Chrome and go to: https://localhost:4173"
echo "Press Ctrl+C to stop"

# Option A: Using npx (if Node.js is installed)
npx serve MLAF/ -l 4173 --ssl

# Option B: Using Python (if only Python is available)
# cd MLAF/
# python3 -m http.server 4173
```

Create a file called `start_mlaf.bat` for Windows machines:

```batch
@echo off
echo Starting MLAF...
echo Open Chrome and go to: http://localhost:4173
echo Press Ctrl+C to stop
npx serve MLAF/ -l 4173
pause
```

### At the Clinic

1. Plug USB drive into the clinic's computer
2. Run `start_mlaf.sh` (Linux/Mac) or `start_mlaf.bat` (Windows)
3. Open Chrome > `http://localhost:4173`
4. Grant camera permission
5. MLAF is running — no internet needed

### Important Notes

- The target machine needs either Node.js or Python installed
- Camera access on `localhost` works without HTTPS
- All data (mastery, phrases, SRS schedules) saves to that machine's localStorage
- To collect session data: ask the clinician to export session reports periodically

### Pros
- Works in locations with zero internet
- Physical delivery — you control exactly who receives it
- Once set up, the clinic can use it independently
- Perfect for government schools in rural West Bengal

### Cons
- Requires Node.js or Python on the target machine
- You cannot update remotely — need to deliver a new USB for updates
- No way to monitor usage remotely

---

## Option 5: Dedicated Tablet Setup (For Long-Term Clinical Trials)

**Best for:** Ongoing clinical trials where one device is dedicated to MLAF at a clinic

**How it works:** You set up a tablet (Android or iPad) with MLAF pre-loaded as a PWA. The clinician uses it daily with patients. You retain all source code.

### Setup

```bash
# Step 1: Build the production bundle
cd "/home/ishu/Desktop/gesture grammar"
npm run build

# Step 2: Serve temporarily on your laptop (connect tablet to same WiFi)
npx vite preview --host 0.0.0.0 --port 4173 --https --cert cert.pem --key key.pem
```

### On the Tablet (Android)

1. Open Chrome on the tablet
2. Navigate to `https://<your-laptop-ip>:4173`
3. Accept the certificate warning
4. Chrome will show "Add to Home Screen" prompt (or tap menu > "Install app")
5. Tap "Install" — MLAF is now a PWA on the tablet's home screen
6. Once installed, MLAF works offline — your laptop is no longer needed
7. The service worker caches all assets locally on the tablet

### On the Tablet (iPad)

1. Open Safari on the iPad
2. Navigate to `https://<your-laptop-ip>:4173`
3. Tap the Share button > "Add to Home Screen"
4. MLAF is now an app icon on the iPad
5. Works offline after initial load

### At the Clinic

- The clinician taps the MLAF icon on the tablet
- It launches like a native app (full screen, no browser chrome)
- Camera access is pre-granted
- All session data persists in the tablet's localStorage
- No internet needed for daily use

### Data Collection

Since all data is in localStorage, you can add a simple export function:

1. At the end of a testing period, visit the tablet
2. Open Chrome DevTools (connect via USB debugging) or use the Session Report export
3. Extract mastery data, automaticity scores, SRS schedules
4. This is your clinical trial data

### Pros
- Feels like a native app to the clinician
- Works offline permanently after installation
- One-time setup, long-term use
- Clinician doesn't need technical knowledge
- All data stays on the tablet (privacy preserved)

### Cons
- Initial setup requires your laptop + same WiFi
- Data collection requires physical visit or USB debugging
- Tablet must have a decent front camera

---

## Recommended Deployment Strategy

| Phase | Option | Purpose |
|---|---|---|
| Kolkata Meeting (March 2026) | Option 1: Local Demo | Live demonstration to neuro experts. Zero IP risk. |
| Immediate Follow-Up | Option 3a: Netlify (password) | Remote experts who want to try it after the meeting. Share URL + password. Revoke after 2 weeks. |
| Clinical Trial Setup | Option 5: Dedicated Tablet | One tablet per participating clinic. You set up, they use. Collect data monthly. |
| Rural School Deployment | Option 4: USB Drive | Government schools without internet. Physical delivery with setup script. |
| Post-Patent Publication | Full public deployment | Once patent is granted, deploy publicly. Open beta. |

---

## Quick Reference: Build Commands

```bash
# Development (with hot reload)
npm run dev:frontend

# Production build
npm run build

# Preview production build locally
npm run preview

# Preview on local network (HTTP)
npx vite preview --host 0.0.0.0 --port 4173

# Preview on local network (HTTPS — needed for camera on non-localhost)
npx vite preview --host 0.0.0.0 --port 4173 --https --cert cert.pem --key key.pem

# Generate self-signed SSL certificate
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 30 -nodes -subj '/CN=mlaf-test'

# Run tests
npm test
```

---

## Checklist: Before Any Demo or Deployment

- [ ] Run `npm run build` — ensure zero errors
- [ ] Test all 13 accessibility profiles in the built version
- [ ] Verify camera access works (HTTPS if non-localhost)
- [ ] Test TTS output (sentence completion speaks aloud)
- [ ] Verify gesture detection works with adequate lighting
- [ ] Clear any personal test data from localStorage if sharing a device
- [ ] Prepare backup demo (screen recording) in case of hardware failure
- [ ] Bring a USB mouse as backup (in case trackpad gesture conflicts)
- [ ] Test on the exact device you will demo on (not just your dev machine)

---

*System Author: Neil Shankar Ray*
*Provisional Patent Filed — MLAF (Multimodal Language Acquisition Framework)*
*Guide prepared: March 2026*
