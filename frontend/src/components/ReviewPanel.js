import React from "react";

function ReviewPanel({ review, loading }) {

  if (loading) {
    return (
      <div className="review">
        <h2>AI Review</h2>
        <p>⏳ Analyzing code...</p>
      </div>
    );
  }

  if (!review) {
    return (
      <div className="review">
        <h2>AI Review</h2>
        <p>No review yet</p>
      </div>
    );
  }

  return (
    <div className="review">

      <h2>AI Review</h2>

      <p><b>Suggestion:</b> {review.suggestion}</p>
      <p><b>Explanation:</b> {review.explanation}</p>
      <p><b>Bug:</b> {review.bug}</p> {/* ✅ FIXED */}
      <p><b>Score:</b> {review.score}</p>

    </div>
  );
}

export default ReviewPanel;