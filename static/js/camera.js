const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture');
    const uploadBtn = document.getElementById('upload');
    const image = document.getElementsByClassName('image-btn');
    const cap = document.getElementsByClassName('capture-btn');
    let capturedBlob = null;

    // Start the camera
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
      })
      .catch(err => {
        alert("❌ Could not access the camera. " + err.message);
        console.error(err);
      });

    // Capture a photo (with object-fit: cover effect)
    captureBtn.addEventListener('click', () => {
      const ctx = canvas.getContext('2d');
      captureBtn.style.display = "";
      video.style.display = "";
      const vWidth = video.videoWidth;
      const vHeight = video.videoHeight;
      const canvasSize = canvas.width;

      let sx, sy, sWidth, sHeight;
      const aspectRatio = vWidth / vHeight;

      if (aspectRatio > 1) {
        sHeight = vHeight;
        sWidth = vHeight;
        sx = (vWidth - sWidth) / 2;
        sy = 0;
      } else {
        sWidth = vWidth;
        sHeight = vWidth;
        sx = 0;
        sy = (vHeight - sHeight) / 2;
      }

      ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, canvasSize, canvasSize);

      canvas.toBlob(blob => {
        capturedBlob = blob;
        alert('✅ Picture captured!');
        image[0].style.display = "block";
        cap[0].style.display = "none";
      }, 'image/png');
    });

    // Upload photo
    uploadBtn.addEventListener('click', () => {
      if (!capturedBlob) return alert('❌ Capture first!');
      const formData = new FormData();
      formData.append('face', new File([capturedBlob], 'face.png', { type: 'image/png' }));
      const selectedWig = document.querySelector('.wig-thumb.selected');
      if (!selectedWig) {
        alert('❌ Please select a wig.');
        return;
      }
      formData.append('shape', selectedWig.getAttribute('data-value'));

      fetch('/get_wig', {
        method: 'POST',
        body: formData
      })
      .then(res => res.json())
      .then(data => {
        alert('✅ Uploaded! Server says: ' + JSON.stringify(data));
      
        const resultImg = document.getElementById('resultImage');
        if (data.image_url) { // assumes your server responds with { image_url: "..." }
          resultImg.src = data.image_url;
          resultImg.style.display = 'block';
        } else {
          alert('❌ No image returned from server.');
        }
      })
      .catch(err => {
        console.error(err);
        alert('❌ Upload failed');
      });
    });

    const wigThumbs = document.querySelectorAll('.wig-thumb');

    wigThumbs.forEach(thumb => {
      thumb.addEventListener('click', () => {
        // Remove "selected" class from all thumbnails
        wigThumbs.forEach(t => t.classList.remove('selected'));
        // Add "selected" class to the clicked one
        thumb.classList.add('selected');
      });
    });

    // Scroll buttons functionality
    document.addEventListener('DOMContentLoaded', function () {
        const container = document.getElementById('wig-container');
        const nextBtn = document.getElementById('wig-next');
        const prevBtn = document.getElementById('wig-prev');
        const scrollStep = 200;
    
        // Adjust button visibility based on scroll position
        function updateButtons() {
            const maxScroll = container.scrollWidth - container.clientWidth;
            prevBtn.style.display = container.scrollLeft > 0 ? 'block' : 'none';
            
            // Ensure that the next button is always shown when the page is first loaded
            nextBtn.style.display = container.scrollLeft < maxScroll || container.scrollLeft === 0 ? 'block' : 'none';
        }

        // When the next button is clicked
        nextBtn.addEventListener('click', () => {
            container.scrollBy({ left: scrollStep, behavior: 'smooth' });
        });

        // When the previous button is clicked
        prevBtn.addEventListener('click', () => {
            container.scrollBy({ left: -scrollStep, behavior: 'smooth' });
        });

        // Update button visibility on scroll
        container.addEventListener('scroll', updateButtons);

        // Initial check to show the next button on page load
        setTimeout(updateButtons, 10);  // Ensure buttons are checked right after loading
    });