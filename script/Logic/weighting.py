from Board.list import list

board = list.getboardset()



class Weights:
    def __init__(self, round):
        self.round = round
        self.weightings = []

    
    def weightrealignment(self):
        if self.round <= 3:
            weights = {
                "ore": 0.8,
                "sheep": 1.0,
                "wood": 1.4,
                "wheat": 1.2,
                "brick": 1.4,
                "desert": 0.0
            }
        # Mid game (rounds 4-7)
        elif self.round <= 7:
            weights = {
                "ore": 1.2,
                "sheep": 1.0,
                "wood": 1.0,
                "wheat": 1.3,
                "brick": 1.0,
                "desert": 0.0
            }
        # Late game (rounds 8+)
        else:
            weights = {
                "ore": 1.4,
                "sheep": 1.0,
                "wood": 0.8,
                "wheat": 1.3,
                "brick": 0.8,
                "desert": 0.0
            }
        
        return weights.get(resource, 0.0)


    def calculate_hex_weights(self):
        """Calculate weighted value for all hexes"""
        self.weightings = []
        
        for i in range(len(self.board)):
            resource_weight = self.get_resource_weight(i)
            probability = self.board[i][2]  # Assuming this is pip count
            
            # Total hex value = resource weight * probability
            hex_value = resource_weight * probability
            
            self.weightings.append(hex_value)
        
        return self.weightings

