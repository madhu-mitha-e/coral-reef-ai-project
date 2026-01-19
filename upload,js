function uploadImage(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  reader.onload = function () {
    load3DImage(reader.result);
    runHealthDetection();
  };

  reader.readAsDataURL(file);
}

function runHealthDetection() {
  const results = [
    { text: "Healthy Coral Reef 🌿", color: "green" },
    { text: "Partially Bleached Coral ⚠️", color: "orange" },
    { text: "Severely Bleached Coral ❌", color: "red" }
  ];

  const result = results[Math.floor(Math.random() * results.length)];

  const box = document.getElementById("analysisResult");
  box.innerText = "Result: " + result.text;
  box.style.color = result.color;
  box.style.fontWeight = "bold";
}
