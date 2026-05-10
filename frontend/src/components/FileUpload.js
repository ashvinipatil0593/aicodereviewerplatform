import React from "react";

function FileUpload({ setCode }) {

  const handleFile = (event) => {

    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = function(e) {
      setCode(e.target.result);
    };

    reader.readAsText(file);
  };

  return (

    <div style={{marginBottom:"20px"}}>

      <label
        style={{
          padding:"8px 14px",
          background:"#2563eb",
          color:"white",
          borderRadius:"6px",
          cursor:"pointer",
          fontSize:"14px"
        }}
      >
        Upload Code File
        <input
          type="file"
          onChange={handleFile}
          style={{display:"none"}}
        />
      </label>

    </div>

  );

}

export default FileUpload;