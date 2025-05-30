// // the variables of the image slider
// const image = document.querySelector(".slide-image");
// const imageNumber = document.querySelector(".slide-image-number");
// const totalImageNumber = document.querySelector(".total-image-number");
// const prevBtn = document.querySelector(".backward");
// const nextBtn = document.querySelector(".forward");

// // the image array
// const images = [
//   "imgs/TOFLogo.png",
//   "imgs/TOFbg.png",
//   "imgs/tof_cover.jpg"
// ];

// // the index of the image on page load
// let currentImage = 0;

// // the image details that shows when the webpage loads
// window.addEventListener("DOMContentLoaded", showImage);

// // function to select and change the image details
// function showImage() {
//   image.src = images[currentImage];
//   imageNumber.textContent = currentImage + 1;
//   totalImageNumber.textContent = images.length;
// }

// // the next button function
// nextBtn.addEventListener("click", function () {
//     currentImage++;
//     if (currentImage > images.length - 1) {
//       currentImage = 0;
//     }
//     showImage(currentImage);
//   });

//   // the prev button function
//   prevBtn.addEventListener("click", function () {
//     currentImage--;
//     if (currentImage < 0) {
//       currentImage = images.length - 1;
//     }
//     showImage(currentImage);
//   });



// let slideIndex = 1;
// showSlides(slideIndex);

// // Next/previous controls
// function plusSlides(n) {
//   showSlides(slideIndex += n);
// }

// // Thumbnail image controls
// function currentSlide(n) {
//   showSlides(slideIndex = n);
// }

// function showSlides(n) {
//   let i;
//   let slides = document.getElementsByClassName("mySlides");
//   let dots = document.getElementsByClassName("dot");
//   if (n > slides.length) {slideIndex = 1}
//   if (n < 1) {slideIndex = slides.length}
//   for (i = 0; i < slides.length; i++) {
//     slides[i].style.display = "none";
//   }
//   for (i = 0; i < dots.length; i++) {
//     dots[i].className = dots[i].className.replace(" active", "");
//   }
//   slides[slideIndex-1].style.display = "block";
//   dots[slideIndex-1].className += " active";
// }


var images = [
    "../imgs/pics/cat1.jpg",
    "../imgs/pics/cat2.jpg",
    "../imgs/pics/cats1.jpg",
    "../imgs/pics/cats2.jpg",
    "../imgs/pics/cover.jpg",
    "../imgs/pics/donkey1.jpg",
    "../imgs/pics/playground.jpg"
];

var count = 0;

function prevClick() {
    var pic = document.getElementById("slidepic");
    var picNum = document.getElementById("slidenum");
    if (count === 0) {
        count = images.length - 1;
    } else {
        count--;
    }
    pic.src = images[count];
    picNum.innerHTML = count + "/" + images.length;
}

function nextClick() {
    var pic = document.getElementById("slidepic");
    var picNum = document.getElementById("slidenum");
    if (count === images.length - 1) {
        count = 0;
    } else {
        count++;
    }
    pic.src = images[count];
    picNum.innerHTML = count + "/" + images.length;
}