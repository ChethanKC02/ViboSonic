const form = document.getElementById("uploadForm");
const statusDiv = document.getElementById("status");
const errorDiv = document.getElementById("error");
const genreChart = document.getElementById("genreChart");
const genreLabelsDiv = document.getElementById("genreLabels");
const resultBox = document.getElementById("resultBox");
const selectedFile = document.getElementById("selectedFile");
const downloadBtn = document.getElementById("downloadBtn");
let chartInstance = null;

document.getElementById("audioFile").addEventListener("change", function () {
  const file = this.files[0];
  if (file) {
    selectedFile.textContent = ` 🎵 Selected File : ${file.name} 🎵`;
    selectedFile.classList.remove("hidden");
  } else {
    selectedFile.classList.add("hidden");
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  statusDiv.classList.remove("hidden");
  errorDiv.classList.add("hidden");
  resultBox.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", document.getElementById("audioFile").files[0]);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    statusDiv.classList.add("hidden");

    if (data.error) {
      errorDiv.textContent = data.error;
      errorDiv.classList.remove("hidden");
    } else {
      const labels = Object.keys(data);
      const values = Object.values(data);
      const colors = [
        "#6366f1", "#3b82f6", "#06b6d4", "#10b981",
        "#84cc16", "#f59e0b", "#ef4444", "#8b5cf6",
        "#ec4899", "#0ea5e9"
      ];

      if (chartInstance) chartInstance.destroy();

      chartInstance = new Chart(genreChart, {
        type: "pie",
        data: {
          labels: labels,
          datasets: [{
            label: "Genre Match %",
            data: values,
            backgroundColor: colors,
            borderColor: "#1f2937",
            borderWidth: 2,
          }]
        },
        options: {
          responsive: true,
          plugins: {
            tooltip: {
              callbacks: {
                label: ctx => `${ctx.label}: ${ctx.raw.toFixed(2)}%`
              }
            },
            legend: { display: false }
          }
        }
      });

      genreLabelsDiv.innerHTML = labels.map((label, i) =>
        `<div class="genre-label-item" style="color:${colors[i]};">
          <span class="genre-dot" style="background:${colors[i]};"></span>${label}
        </div>`).join("");

      resultBox.classList.remove("hidden");
      downloadBtn.classList.remove("hidden");
    }

  } catch (err) {
    statusDiv.classList.add("hidden");
    errorDiv.textContent = "An error occurred. Please try again.";
    errorDiv.classList.remove("hidden");
  }
});

downloadBtn.addEventListener("click", async () => {
  const { jsPDF } = window.jspdf;
  const fileInput = document.getElementById("audioFile");
  const fileName = fileInput.files[0]?.name || "Unknown File";

  const pdf = new jsPDF("p", "mm", "a4");
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();

  pdf.setLineWidth(1);
  pdf.setDrawColor(180);
  pdf.rect(10, 10, pageWidth - 20, pageHeight - 20);

  pdf.setLineWidth(0.2);
  pdf.setFont("Helvetica", "bold");
  pdf.setFontSize(18);
  pdf.text("ViboSonic Genre Prediction Report", pageWidth / 2, 20, { align: "center" });

  pdf.setDrawColor(200);
  pdf.line(20, 24, pageWidth - 20, 24);

  pdf.setFont("Helvetica", "normal");
  pdf.setFontSize(12);
  pdf.text("Audio File:", 20, 32);
  pdf.setFont("Helvetica", "bold");
  pdf.text(fileName, 45, 32);

  pdf.line(20, 36, pageWidth - 20, 36);

  const chartCanvas = document.getElementById("genreChart");
  const chartImage = chartCanvas.toDataURL("image/png", 1.0);
  const chartWidth = 90;
  const chartHeight = 90;
  pdf.addImage(chartImage, "PNG", (pageWidth - chartWidth) / 2, 42, chartWidth, chartHeight);

  let y = 42 + chartHeight + 8;
  pdf.setDrawColor(220);
  pdf.line(20, y, pageWidth - 20, y);
  y += 6;

  pdf.setFont("Helvetica", "bold");
  pdf.setFontSize(13);
  pdf.text("Genre Prediction Breakdown", 20, y);
  y += 7;

  pdf.setFont("Helvetica", "normal");
  pdf.setFontSize(11);
  const description = "This section shows the predicted genres for your uploaded music file, visualized through a pie chart and listed below with their associated confidence percentages.";
  const splitDescription = pdf.splitTextToSize(description, pageWidth - 40);
  pdf.text(splitDescription, 20, y);
  y += splitDescription.length * 6 + 4;

  const chart = Chart.getChart("genreChart");
  const labels = chart.data.labels;
  const data = chart.data.datasets[0].data;
  const colors = chart.data.datasets[0].backgroundColor;

  pdf.setFontSize(11);
  pdf.setTextColor(0);

  for (let i = 0; i < labels.length; i++) {
    if (y > pageHeight - 20) {
      pdf.addPage();
      y = 20;
    }

    pdf.setFillColor(colors[i]);
    pdf.circle(22, y - 2, 2, "F");

    pdf.setFont("Helvetica", "normal");
    pdf.text(`${labels[i]}`, 28, y);

    pdf.setFont("Helvetica", "bold");
    pdf.text(`${data[i].toFixed(2)}%`, pageWidth - 20, y, { align: "right" });

    y += 7.2;
  }

  pdf.save("ViboSonic_Genre_Report.pdf");
});
