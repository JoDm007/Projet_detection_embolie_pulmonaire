// FUMPE AI 3D — Landing Page Interactive Scripts & CT Simulator

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. Hero Canvas CT Slice Preview Animation ---
    const heroCanvas = document.getElementById('heroCanvas');
    if (heroCanvas) {
        const ctx = heroCanvas.getContext('2d');
        let frame = 0;

        function renderHeroSlice() {
            ctx.fillStyle = '#050811';
            ctx.fillRect(0, 0, heroCanvas.width, heroCanvas.height);

            // Draw simulated lung contour 3D
            const centerX = heroCanvas.width / 2;
            const centerY = heroCanvas.height / 2;

            // Thoracic cavity outline
            ctx.beginPath();
            ctx.ellipse(centerX, centerY, 150, 110, 0, 0, 2 * Math.PI);
            ctx.strokeStyle = '#1E293B';
            ctx.lineWidth = 4;
            ctx.stroke();

            // Left Lung
            ctx.beginPath();
            ctx.ellipse(centerX - 60, centerY, 55, 80, -0.2, 0, 2 * Math.PI);
            ctx.fillStyle = '#0B132B';
            ctx.fill();
            ctx.strokeStyle = '#334155';
            ctx.stroke();

            // Right Lung
            ctx.beginPath();
            ctx.ellipse(centerX + 60, centerY, 55, 80, 0.2, 0, 2 * Math.PI);
            ctx.fillStyle = '#0B132B';
            ctx.fill();
            ctx.strokeStyle = '#334155';
            ctx.stroke();

            // Pulmonary Arteries (High Contrast Blood Vessels - White/Light Cyan)
            ctx.beginPath();
            ctx.arc(centerX - 40, centerY - 10, 18, 0, 2 * Math.PI);
            ctx.arc(centerX + 40, centerY - 10, 20, 0, 2 * Math.PI);
            ctx.fillStyle = '#E2E8F0';
            ctx.fill();

            // Animated PE Blood Clot Overlay (Red Segmented Region)
            const pulseSize = 10 + Math.sin(frame * 0.05) * 2;
            ctx.beginPath();
            ctx.arc(centerX + 40, centerY - 10, pulseSize, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(255, 59, 92, 0.75)';
            ctx.fill();
            ctx.strokeStyle = '#FF3B5C';
            ctx.lineWidth = 2;
            ctx.stroke();

            // 3D Voxel mesh lines
            ctx.strokeStyle = 'rgba(6, 182, 212, 0.15)';
            ctx.lineWidth = 1;
            for (let i = 0; i < heroCanvas.width; i += 30) {
                ctx.beginPath();
                ctx.moveTo(i, 0);
                ctx.lineTo(i, heroCanvas.height);
                ctx.stroke();
            }

            frame++;
            requestAnimationFrame(renderHeroSlice);
        }

        renderHeroSlice();
    }

    // --- 2. Interactive CT Scan Simulator ---
    const simCanvas = document.getElementById('simCanvas');
    const sliceRange = document.getElementById('sliceRange');
    const sliceNumVal = document.getElementById('sliceNumVal');
    const btnModeOverlay = document.getElementById('btnModeOverlay');
    const btnModeRaw = document.getElementById('btnModeRaw');
    const huVal = document.getElementById('huVal');
    const volVal = document.getElementById('volVal');
    const confVal = document.getElementById('confVal');
    const simTooltip = document.getElementById('simTooltip');

    let showOverlay = true;

    if (simCanvas && sliceRange) {
        const sCtx = simCanvas.getContext('2d');

        function drawCTSlice(sliceNum) {
            const w = simCanvas.width;
            const h = simCanvas.height;
            const cx = w / 2;
            const cy = h / 2;

            // Background DICOM canvas
            sCtx.fillStyle = '#05070E';
            sCtx.fillRect(0, 0, w, h);

            // Body contour
            sCtx.beginPath();
            sCtx.ellipse(cx, cy, 210, 160, 0, 0, 2 * Math.PI);
            sCtx.fillStyle = '#0F172A';
            sCtx.fill();
            sCtx.strokeStyle = '#1E293B';
            sCtx.lineWidth = 3;
            sCtx.stroke();

            // Spine / Vertebra (Bone HU > 500)
            sCtx.beginPath();
            sCtx.arc(cx, cy + 130, 25, 0, 2 * Math.PI);
            sCtx.fillStyle = '#F8FAFC';
            sCtx.fill();

            // Left Lung cavity (-800 HU)
            sCtx.beginPath();
            sCtx.ellipse(cx - 95, cy - 10, 80, 110, -0.15, 0, 2 * Math.PI);
            sCtx.fillStyle = '#040814';
            sCtx.fill();
            sCtx.strokeStyle = '#1E293B';
            sCtx.stroke();

            // Right Lung cavity (-800 HU)
            sCtx.beginPath();
            sCtx.ellipse(cx + 95, cy - 10, 80, 110, 0.15, 0, 2 * Math.PI);
            sCtx.fillStyle = '#040814';
            sCtx.fill();
            sCtx.strokeStyle = '#1E293B';
            sCtx.stroke();

            // Pulmonary Trunk & Arteries (+150 HU Contrast)
            sCtx.beginPath();
            sCtx.arc(cx - 60, cy - 20, 26, 0, 2 * Math.PI);
            sCtx.arc(cx + 60, cy - 20, 28, 0, 2 * Math.PI);
            sCtx.fillStyle = '#CBD5E1';
            sCtx.fill();

            // Slice variation: clot appears around slices 100 - 180
            const hasClotInSlice = sliceNum >= 90 && sliceNum <= 210;

            if (hasClotInSlice) {
                // Clot region inside right artery (HU = -10 to +30)
                const clotRadius = Math.sin((sliceNum - 90) / 120 * Math.PI) * 18 + 4;

                // Dark Filling Defect (Raw Scanner View)
                sCtx.beginPath();
                sCtx.arc(cx + 60, cy - 20, clotRadius, 0, 2 * Math.PI);
                sCtx.fillStyle = '#334155'; // Dark filling defect
                sCtx.fill();

                // If Overlay is ON, draw AI 3D Voxel Segmentation (Red/Cyan Highlight)
                if (showOverlay) {
                    sCtx.beginPath();
                    sCtx.arc(cx + 60, cy - 20, clotRadius + 2, 0, 2 * Math.PI);
                    sCtx.fillStyle = 'rgba(255, 59, 92, 0.7)';
                    sCtx.fill();
                    sCtx.strokeStyle = '#FF3B5C';
                    sCtx.lineWidth = 3;
                    sCtx.stroke();

                    // Voxel grid overlay
                    sCtx.fillStyle = 'rgba(6, 182, 212, 0.4)';
                    sCtx.font = '10px monospace';
                    sCtx.fillText('VOXEL 3D: [124, 210, ' + sliceNum + ']', cx + 85, cy - 25);
                }

                // Update Metrics Display
                const calculatedVolume = Math.round(1420 * Math.sin((sliceNum - 90) / 120 * Math.PI));
                volVal.textContent = calculatedVolume + ' mm³';
                confVal.textContent = (94.2 + (sliceNum % 5) * 0.8).toFixed(1) + '%';
                huVal.textContent = '-12 HU (Defaut de remplissage)';
                huVal.className = 'sm-val text-red';
            } else {
                volVal.textContent = '0 mm³';
                confVal.textContent = 'N/A';
                huVal.textContent = '+140 HU (Tissu San opacifié)';
                huVal.className = 'sm-val text-cyan';
            }
        }

        // Event Listeners for Simulator
        sliceRange.addEventListener('input', (e) => {
            const val = e.target.value;
            sliceNumVal.textContent = val;
            drawCTSlice(parseInt(val));
        });

        if (btnModeOverlay && btnModeRaw) {
            btnModeOverlay.addEventListener('click', () => {
                showOverlay = true;
                btnModeOverlay.classList.add('active');
                btnModeRaw.classList.remove('active');
                drawCTSlice(parseInt(sliceRange.value));
            });

            btnModeRaw.addEventListener('click', () => {
                showOverlay = false;
                btnModeRaw.classList.add('active');
                btnModeOverlay.classList.remove('active');
                drawCTSlice(parseInt(sliceRange.value));
            });
        }

        // Interactive Canvas Hover Tooltip
        simCanvas.addEventListener('mousemove', (e) => {
            const rect = simCanvas.getBoundingClientRect();
            const x = Math.round(e.clientX - rect.left);
            const y = Math.round(e.clientY - rect.top);

            simTooltip.style.display = 'block';
            simTooltip.style.left = (x + 15) + 'px';
            simTooltip.style.top = (y + 15) + 'px';

            // Calculate mock HU based on distance to center
            const dist = Math.hypot(x - 240, y - 240);
            let HU = -1000;
            if (dist < 100) HU = -800; // Lung
            if (dist < 30) HU = 150;  // Vessel

            simTooltip.innerHTML = `Voxel (${x}, ${y})<br><strong>${HU} HU</strong>`;
        });

        simCanvas.addEventListener('mouseleave', () => {
            simTooltip.style.display = 'none';
        });

        // Initial Draw
        drawCTSlice(parseInt(sliceRange.value));
    }

    // --- 3. Tab Switcher for AI Architecture ---
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(target).classList.add('active');
        });
    });
});
