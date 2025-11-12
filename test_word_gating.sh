#!/bin/bash

# Word Gating System Test Script
# This script tests the complete admin word gating workflow

API_URL="http://localhost:9000"

echo "=========================================="
echo "🔐 WORD GATING SYSTEM TEST"
echo "=========================================="
echo ""

echo "1️⃣  Testing Admin Open Word Endpoint..."
echo "Opening HELLO for contributions..."
curl -s -X POST "$API_URL/api/ama/words/HELLO/open"
echo -e "\n"

echo "Opening PLEASE for contributions..."
curl -s -X POST "$API_URL/api/ama/words/PLEASE/open"
echo -e "\n"

echo "Opening WATER for contributions..."
curl -s -X POST "$API_URL/api/ama/words/WATER/open"
echo -e "\n"

echo "2️⃣  Testing Public Word List (should show only open words)..."
curl -s "$API_URL/api/dictionary-words?page=1&per_page=10"
echo -e "\n\n"

echo "3️⃣  Testing Admin Word List (should show all words with is_open status)..."
curl -s "$API_URL/api/ama/words?limit=5"
echo -e "\n\n"

echo "4️⃣  Testing Close Word Endpoint..."
echo "Closing WATER..."
curl -s -X POST "$API_URL/api/ama/words/WATER/close"
echo -e "\n"

echo "5️⃣  Verifying WATER is removed from public list..."
curl -s "$API_URL/api/dictionary-words?page=1&per_page=10"
echo -e "\n\n"

echo "6️⃣  Testing Bulk Open Endpoint..."
echo "Opening FAMILY, SCHOOL, FRIEND..."
curl -s -X POST "$API_URL/api/ama/words/bulk-open" \
  -H "Content-Type: application/json" \
  -d '["FAMILY", "SCHOOL", "FRIEND"]'
echo -e "\n"

echo "7️⃣  Final Public Word List Check..."
curl -s "$API_URL/api/dictionary-words?page=1&per_page=10"
echo -e "\n\n"

echo "=========================================="
echo "✅ WORD GATING SYSTEM TEST COMPLETE"
echo "=========================================="
