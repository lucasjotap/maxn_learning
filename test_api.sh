#!/bin/bash
# Test API endpoints script

BASE_URL="${1:-http://localhost:8000}"

echo "🧪 Testing Recommendation Engine API"
echo "Base URL: $BASE_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4
    
    echo -e "${BLUE}Testing:${NC} $description"
    echo "  $method $endpoint"
    
    if [ -n "$data" ]; then
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X "$method" \
            "$BASE_URL$endpoint")
    fi
    
    http_code=$(echo "$response" | grep HTTP_STATUS | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_STATUS/d')
    
    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        echo -e "${GREEN}✓ Success${NC} (HTTP $http_code)"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body" | head -20
    else
        echo -e "✗ Failed (HTTP $http_code)"
        echo "$body" | head -10
    fi
    echo ""
}

# Health check
test_endpoint "GET" "/health" "" "Health Check"

# Get user stats
test_endpoint "GET" "/users/user1/stats" "" "Get User Stats"

# Get daily playlist
test_endpoint "GET" "/automation/daily-playlist/user1" "" "Daily Playlist"

# Get weekly movies
test_endpoint "GET" "/automation/weekly-movies/user1" "" "Weekly Movies"

# Get task prioritization
test_endpoint "GET" "/automation/task-prioritization/user1" "" "Task Prioritization"

# Get recommendations
test_endpoint "POST" "/recommendations" \
    '{
        "user_id": "user1",
        "domain": "movies",
        "top_k": 5
    }' \
    "Get Movie Recommendations"

# Log interaction
test_endpoint "POST" "/interactions" \
    '{
        "user_id": "user1",
        "domain": "movies",
        "item_id": "movie_1",
        "interaction_type": "like",
        "rating": 5.0
    }' \
    "Log Interaction"

# Get training stats
test_endpoint "GET" "/training/stats" "" "Training Statistics"

echo "✅ API testing complete!"
echo ""
echo "💡 View interactive API docs at: $BASE_URL/docs"

