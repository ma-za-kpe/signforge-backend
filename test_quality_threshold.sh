#!/bin/bash

# Quality Threshold Test Script
# Tests that contributions below 60% quality are rejected

API_URL="http://localhost:9000"

echo "=========================================="
echo "🎯 QUALITY THRESHOLD TEST (60% minimum)"
echo "=========================================="
echo ""

echo "This test simulates a low-quality contribution"
echo "Expected: API should reject with 422 error"
echo ""

# Create a test contribution with intentionally poor quality
# This uses minimal frames with poor hand detection

TEST_PAYLOAD=$(cat <<'EOF'
{
  "word": "TEST",
  "frames": [
    {
      "landmarks": [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3},
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.3}
      ],
      "timestamp": 0
    }
  ],
  "user_id": "test-quality-threshold",
  "duration": 0.5,
  "timestamp": "2025-01-01T00:00:00Z"
}
EOF
)

echo "📤 Sending low-quality contribution..."
echo ""

RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$API_URL/api/contribute" \
  -H "Content-Type: application/json" \
  -d "$TEST_PAYLOAD")

HTTP_BODY=$(echo "$RESPONSE" | sed -e 's/HTTP_STATUS\:.*//g')
HTTP_STATUS=$(echo "$RESPONSE" | tr -d '\n' | sed -e 's/.*HTTP_STATUS://')

echo "📥 Response:"
echo "$HTTP_BODY"
echo ""
echo "Status Code: $HTTP_STATUS"
echo ""

if [ "$HTTP_STATUS" == "422" ]; then
    echo "✅ TEST PASSED: Low-quality contribution was rejected (422)"
    echo ""
    echo "Quality threshold is working correctly!"
else
    echo "❌ TEST FAILED: Expected 422, got $HTTP_STATUS"
    echo ""
    echo "The contribution should have been rejected."
fi

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
