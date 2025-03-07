function postSentimen(e) {
    var sheet = e.source.getActiveSheet();
    if (sheet.getName() !== "Sentimen") return;
  
    var range = e.range;
    var col = range.getColumn();
    if (col !== 1) return; // Hanya memproses kolom A
  
    var values = range.getValues();
    var rows = range.getNumRows();
    var startRow = range.getRow();
  
    var { payloads, rowsToClear } = prepareDataForRequest(values, startRow);
  
    // Hapus isi kolom B jika kolom A kosong
    clearEmptySentimentRows(sheet, rowsToClear);
  
    // Kirim data ke server jika ada input
    if (payloads.length > 0) {
      sendSentimentRequest(payloads, sheet);
    }
  }
  
  /**
   * Mempersiapkan data sebelum dikirim ke server
   * @param {Array} values - Nilai dari sel yang dipilih
   * @param {Number} startRow - Baris awal data
   * @returns {Object} - Objek dengan array payloads dan rowsToClear
   */
  function prepareDataForRequest(values, startRow) {
    var payloads = [];
    var rowsToClear = [];
  
    for (var i = 0; i < values.length; i++) {
      if (values[i][0] === null || values[i][0] === "") {
        rowsToClear.push(startRow + i);
      } else {
        payloads.push({
          "row": startRow + i,
          "komentar": values[i][0]
        });
      }
    }
  
    return { payloads, rowsToClear };
  }
  
  /**
   * Menghapus data di kolom B jika kolom A kosong
   * @param {Object} sheet - Objek sheet Google Spreadsheet
   * @param {Array} rowsToClear - Daftar baris yang harus dikosongkan
   */
  function clearEmptySentimentRows(sheet, rowsToClear) {
    rowsToClear.forEach(function (row) {
      sheet.getRange(row, 2).setValue("");
    });
  }
  
  /**
   * Mengirim request ke server untuk analisis sentimen
   * @param {Array} payloads - Data komentar yang akan dikirim
   * @param {Object} sheet - Objek sheet Google Spreadsheet
   */
  function sendSentimentRequest(payloads, sheet) {
    var url = "https://7stm1xg6-5000.asse.devtunnels.ms/predict";
  
    var options = {
      "method": "post",
      "contentType": "application/json",
      "payload": JSON.stringify({ "data": payloads })
    };
  
    try {
      var response = UrlFetchApp.fetch(url, options);
      var jsonResponse = JSON.parse(response.getContentText());
      console.log(jsonResponse);
  
      if (jsonResponse && jsonResponse.data) {
        jsonResponse.data.forEach(function (item) {
          sheet.getRange(item.row, 2).setValue(item.sentimen);
        });
      }
    } catch (error) {
      console.error("Gagal mengirim data: " + error.message);
    }
  }
  