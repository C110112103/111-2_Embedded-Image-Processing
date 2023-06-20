
var button = document.getElementById("abutton");

button.addEventListener("click", function () {
    alert("This is what will happen after clicking!")
})

var button2 = document.getElementById("show_school_btn");

button2.addEventListener("click", function () {
    window.location.href = "http://127.0.0.1:8000/get_test/";
})