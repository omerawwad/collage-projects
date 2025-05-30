var cats = [
    "../imgs/pics/cat1.jpg",
    "../imgs/pics/cat2.jpg",
    "../imgs/pics/cats1.jpg",
    "../imgs/pics/cats2.jpg",
];

var donkeys = [
    "../imgs/pics/donkey1.jpg",
];

var arr_name = [];
arr_name = cats;

function set_array() {
    arr_name = donkeys;
}

var count = 0;

function prevClick() {
    var pic = document.getElementById("slidepic");
    var picNum = document.getElementById("slidenum");
    if (count === 0) {
        count = arr_name.length - 1;
    } else {
        count--;
    }
    pic.src = arr_name[count];
    picNum.innerHTML = count + "/" + arr_name.length;
}

function nextClick() {
    var pic = document.getElementById("slidepic");
    var picNum = document.getElementById("slidenum");
    if (count === arr_name.length - 1) {
        count = 0;
    } else {
        count++;
    }
    pic.src = arr_name[count];
    picNum.innerHTML = count + "/" + arr_name.length;
}