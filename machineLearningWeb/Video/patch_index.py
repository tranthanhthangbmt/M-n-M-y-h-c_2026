import os
import re

base_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\Video'

new_showSlide = """    async function showSlide(index) {
      // Validate index
      if (index < 1) index = 1;
      if (index > TOTAL_SLIDES) index = TOTAL_SLIDES;
      currentSlideIndex = index;

      // Stop current playback
      slideAudio.pause();
      cancelAnimationFrame(scrollAnimationId);
      showLoading(true, `Đang tải slide ${index}...`);

      // Update nút điều hướng
      prevBtn.disabled = (index === 1);
      nextBtn.disabled = (index === TOTAL_SLIDES);
      slideCounter.textContent = `${index} / ${TOTAL_SLIDES}`;

      // Media Session API cho Lock Screen
      if ('mediaSession' in navigator) {
        navigator.mediaSession.metadata = new MediaMetadata({
          title: `Slide ${index} / ${TOTAL_SLIDES}`,
          artist: 'Bài giảng'
        });
      }

      try {
        // 1. Gán source Audio và phát NGAY LẬP TỨC trên thẻ <audio> chính.
        // Tuyệt đối không await việc tải Ảnh ở đây, vì khi tắt màn hình, 
        // trình duyệt (đặc biệt iOS/Android) sẽ ngưng tải Image khiến Audio không bao giờ được play.
        slideAudio.src = `${AUDIO_PREFIX}${index}${AUD_EXT}`;
        
        slideAudio.play().then(() => {
          startAutoScrollIfNeeded();
        }).catch(err => {
          console.log("Autoplay blocked or waiting for interaction", err);
          playPauseBtn.innerHTML = '▶️'; 
        });

        // 2. Tải ảnh chạy ngầm, xong lúc nào thì hiển thị lúc đó
        resourceManager.loadImage(index).then(() => {
          slideImage.src = `${IMAGE_PREFIX}${index}${IMG_EXT}`;
          slideViewer.scrollTop = 0;
          showLoading(false);
        }).catch(err => {
          console.warn("Lỗi tải ảnh:", err);
          showLoading(false);
        });

        // 3. Vẫn tải audio vào cache manager để tương thích
        resourceManager.loadAudio(index).catch(e => {});

        // 4. Preload các slide tiếp theo
        preloadNextSlides(index);

      } catch (error) {
        console.error(error);
        showLoading(true, "Mạng không ổn định. Đang thử lại...");
        setTimeout(() => showSlide(index), 3000);
      }
    }"""

new_init = """    /* ================= INIT ================= */
    window.addEventListener('load', () => {
      loadVideoLinks(); // Tải link video
      showControlsAndResetTimer();
      syncAutoscrollUI();
      checkOrientation(); // Kiểm tra hướng ngay khi tải
      
      // Khởi tạo Media Session Action Handlers
      if ('mediaSession' in navigator) {
        navigator.mediaSession.setActionHandler('previoustrack', () => {
          if (currentSlideIndex > 1) showSlide(currentSlideIndex - 1);
        });
        navigator.mediaSession.setActionHandler('nexttrack', () => {
          if (currentSlideIndex < TOTAL_SLIDES) showSlide(currentSlideIndex + 1);
        });
      }
      
      // Bắt đầu
      showSlide(currentSlideIndex);
    });"""

for i in range(1, 30):
    folder = f'Chapter_{i:02d}'
    filepath = os.path.join(base_dir, folder, 'index.html')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Already updated?
        if "Media Session API cho Lock Screen" in content:
            print(f"Skipping {folder}, already updated.")
            continue
            
        # Replace showSlide
        showSlide_pattern = re.compile(r'async function showSlide\(index\) \{.*?(?=function preloadNextSlides)', re.DOTALL)
        content = showSlide_pattern.sub(new_showSlide + '\n\n    ', content)
        
        # Replace INIT
        init_pattern = re.compile(r'/\* ================= INIT ================= \*/.*?\}\);', re.DOTALL)
        content = init_pattern.sub(new_init, content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {folder}")
