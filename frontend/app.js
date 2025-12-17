async function analyze() {
  const history = JSON.parse(document.getElementById("input").value);

  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({history})
  });

  const data = await res.json();
  document.getElementById("result").textContent =
    JSON.stringify(data, null, 2);

console.log(payload)

}
