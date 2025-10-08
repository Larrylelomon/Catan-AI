import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

class CatanBoardDetector:
    """Main class that orchestrates all detection tasks"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize all models
        self.hex_classifier = HexResourceClassifier().to(self.device)
        self.number_classifier = NumberTokenClassifier().to(self.device)
        self.piece_detector = GamePieceDetector().to(self.device)
        
        # Load trained weights when available
        # self.load_models()
        
        self.board_structure = None
        
    def analyze_board(self, image_path):
        """Main pipeline - analyze entire board"""
        img = cv2.imread(image_path)
        
        # Step 1: Find hex positions (using classical CV)
        hex_positions = self.detect_hex_positions(img)
        
        # Step 2: Classify each hex resource type
        hex_resources = self.classify_hex_resources(img, hex_positions)
        
        # Step 3: Find and classify number tokens
        number_tokens = self.detect_number_tokens(img, hex_positions)
        
        # Step 4: Detect all game pieces (settlements, cities, roads)
        game_pieces = self.detect_game_pieces(img, hex_positions)
        
        # Step 5: Build complete board state
        board_state = self.build_board_state(
            hex_positions, 
            hex_resources, 
            number_tokens, 
            game_pieces
        )
        
        return board_state
    
    def detect_hex_positions(self, img):
        """Use classical CV to find all 19 hexagon positions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hex_positions = []
        for contour in contours:
            # Approximate the contour
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Hexagons have 6 sides
            if len(approx) == 6:
                # Get bounding box and center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Get bounding rectangle for cropping
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    hex_positions.append({
                        'center': (cx, cy),
                        'contour': approx,
                        'bbox': (x, y, w, h),
                        'vertices': self.get_hex_vertices(approx)
                    })
        
        # Sort hexes by position (top to bottom, left to right)
        hex_positions = sorted(hex_positions, key=lambda h: (h['center'][1], h['center'][0]))
        
        return hex_positions
    
    def get_hex_vertices(self, approx):
        """Extract the 6 vertices of the hexagon for road/settlement detection"""
        vertices = []
        for point in approx:
            vertices.append((point[0][0], point[0][1]))
        return vertices
    
    def classify_hex_resources(self, img, hex_positions):
        """Use CNN to classify resource type for each hex"""
        resources = []
        
        for hex_info in hex_positions:
            x, y, w, h = hex_info['bbox']
            
            # Crop the hex region
            hex_crop = img[y:y+h, x:x+w]
            
            # Classify using CNN
            resource_type = self.classify_single_hex(hex_crop)
            resources.append(resource_type)
        
        return resources
    
    def classify_single_hex(self, hex_crop):
        """CNN classification for a single hex"""
        pil_img = Image.fromarray(cv2.cvtColor(hex_crop, cv2.COLOR_BGR2RGB))
        tensor = hex_transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.hex_classifier(tensor)
            pred = torch.argmax(output, dim=1).item()
        
        resource_map = {
            0: 'wood', 
            1: 'brick', 
            2: 'wheat', 
            3: 'sheep', 
            4: 'ore', 
            5: 'desert'
        }
        return resource_map[pred]
    
    def detect_number_tokens(self, img, hex_positions):
        """Find and classify number tokens on each hex"""
        numbers = []
        
        for hex_info in hex_positions:
            cx, cy = hex_info['center']
            
            # Look for circular token in center of hex
            token_region = self.extract_center_region(img, cx, cy, radius=30)
            
            # Classify the number
            number = self.classify_number_token(token_region)
            numbers.append(number)
        
        return numbers
    
    def extract_center_region(self, img, cx, cy, radius):
        """Extract circular region from hex center"""
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(img.shape[1], cx + radius)
        y2 = min(img.shape[0], cy + radius)
        
        return img[y1:y2, x1:x2]
    
    def classify_number_token(self, token_crop):
        """CNN classification for number tokens"""
        # Could also use OCR (Tesseract) as alternative
        pil_img = Image.fromarray(cv2.cvtColor(token_crop, cv2.COLOR_BGR2RGB))
        tensor = number_transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.number_classifier(tensor)
            pred = torch.argmax(output, dim=1).item()
        
        # Map to actual numbers (excluding 7)
        number_map = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8, 
                     6: 9, 7: 10, 8: 11, 9: 12, 10: None}
        return number_map[pred]
    
    def detect_game_pieces(self, img, hex_positions):
        """Detect settlements, cities, and roads"""
        pieces = {
            'settlements': [],
            'cities': [],
            'roads': []
        }
        
        # Get all vertices and edges from hexes
        all_vertices = self.get_all_vertices(hex_positions)
        all_edges = self.get_all_edges(hex_positions)
        
        # Detect settlements at vertices
        pieces['settlements'] = self.detect_settlements(img, all_vertices)
        
        # Detect cities at vertices
        pieces['cities'] = self.detect_cities(img, all_vertices)
        
        # Detect roads on edges
        pieces['roads'] = self.detect_roads(img, all_edges)
        
        return pieces
    
    def get_all_vertices(self, hex_positions):
        """Extract all unique vertex positions from hexes"""
        vertices = []
        seen = set()
        
        for hex_info in hex_positions:
            for vertex in hex_info['vertices']:
                # Round to nearest pixel to group nearby vertices
                rounded = (round(vertex[0]), round(vertex[1]))
                if rounded not in seen:
                    seen.add(rounded)
                    vertices.append(vertex)
        
        return vertices
    
    def get_all_edges(self, hex_positions):
        """Extract all edge positions (between vertices)"""
        edges = []
        
        for hex_info in hex_positions:
            vertices = hex_info['vertices']
            for i in range(len(vertices)):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % len(vertices)]
                edges.append((v1, v2))
        
        return edges
    
    def detect_settlements(self, img, vertices):
        """Detect settlements at vertex positions"""
        settlements = []
        
        for vertex in vertices:
            x, y = vertex
            
            # Extract small region around vertex
            region = self.extract_region(img, x, y, size=20)
            
            # Detect if there's a settlement (colored house shape)
            has_settlement, color = self.detect_piece_at_location(region, piece_type='settlement')
            
            if has_settlement:
                settlements.append({
                    'position': vertex,
                    'player_color': color
                })
        
        return settlements
    
    def detect_cities(self, img, vertices):
        """Detect cities at vertex positions"""
        cities = []
        
        for vertex in vertices:
            x, y = vertex
            region = self.extract_region(img, x, y, size=25)
            
            has_city, color = self.detect_piece_at_location(region, piece_type='city')
            
            if has_city:
                cities.append({
                    'position': vertex,
                    'player_color': color
                })
        
        return cities
    
    def detect_roads(self, img, edges):
        """Detect roads on edges"""
        roads = []
        
        for edge in edges:
            (x1, y1), (x2, y2) = edge
            
            # Extract line region
            line_region = self.extract_line_region(img, x1, y1, x2, y2)
            
            has_road, color = self.detect_piece_at_location(line_region, piece_type='road')
            
            if has_road:
                roads.append({
                    'edge': edge,
                    'player_color': color
                })
        
        return roads
    
    def extract_region(self, img, cx, cy, size):
        """Extract square region around a point"""
        x1 = max(0, int(cx - size))
        y1 = max(0, int(cy - size))
        x2 = min(img.shape[1], int(cx + size))
        y2 = min(img.shape[0], int(cy + size))
        
        return img[y1:y2, x1:x2]
    
    def extract_line_region(self, img, x1, y1, x2, y2):
        """Extract region along a line (for road detection)"""
        # Create a mask for the line with some thickness
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=10)
        
        # Extract pixels along the line
        return cv2.bitwise_and(img, img, mask=mask)
    
    def detect_piece_at_location(self, region, piece_type):
        """Detect if a game piece exists and determine its color"""
        if region.size == 0:
            return False, None
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different players
        player_colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'orange': ([10, 100, 100], [25, 255, 255])
        }
        
        for color_name, (lower, upper) in player_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Check if enough pixels match this color
            pixel_count = cv2.countNonZero(mask)
            threshold = 20 if piece_type == 'road' else 50
            
            if pixel_count > threshold:
                return True, color_name
        
        return False, None
    
    def build_board_state(self, hex_positions, resources, numbers, pieces):
        """Combine all detected information into unified board state"""
        board_state = {
            'hexes': [],
            'settlements': pieces['settlements'],
            'cities': pieces['cities'],
            'roads': pieces['roads']
        }
        
        for i, hex_info in enumerate(hex_positions):
            board_state['hexes'].append({
                'index': i,
                'position': hex_info['center'],
                'resource': resources[i],
                'number': numbers[i],
                'vertices': hex_info['vertices']
            })
        
        return board_state


# CNN Models for classification

class HexResourceClassifier(nn.Module):
    """Classifies hex tiles into resource types"""
    def __init__(self, num_classes=6):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class NumberTokenClassifier(nn.Module):
    """Classifies number tokens (2-12 excluding 7)"""
    def __init__(self, num_classes=11):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class GamePieceDetector(nn.Module):
    """Detects and classifies game pieces (settlements, cities, roads)"""
    def __init__(self, num_classes=4):  # settlement, city, road, none
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# Data transforms
hex_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

number_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Usage example
if __name__ == "__main__":
    detector = CatanBoardDetector()
    
    # Analyze a board image
    board_state = detector.analyze_board('catan_board.jpg')
    
    print("Detected Board State:")
    print(f"Number of hexes: {len(board_state['hexes'])}")
    print(f"Settlements: {len(board_state['settlements'])}")
    print(f"Cities: {len(board_state['cities'])}")
    print(f"Roads: {len(board_state['roads'])}")
    
    # Example: Print first hex info
    if board_state['hexes']:
        hex0 = board_state['hexes'][0]
        print(f"\nFirst hex:")
        print(f"  Resource: {hex0['resource']}")
        print(f"  Number: {hex0['number']}")
        print(f"  Position: {hex0['position']}")