class ModelConfig:
    """Class that represents the model configuration."""
    def __init__(self, model_type, confidence, conf_detection, conf_pose, model_path):
        self.model_type = model_type
        self.confidence = confidence
        self.conf_detection = conf_detection
        self.conf_pose = conf_pose
        self.model_path = model_path

    def get_model_type(self):
        return self.model_type

    def get_confidence(self):
        return self.confidence

    def get_conf_detection(self):
        return self.conf_detection

    def get_conf_pose(self):
        return self.conf_pose

    def get_model_path(self):
        return self.model_path

    def set_model_type(self, model_type):
        self.model_type = model_type

    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_conf_detection(self, conf_detection):
        self.conf_detection = conf_detection

    def set_conf_pose(self, conf_pose):
        self.conf_pose = conf_pose

    def set_model_path(self, model_path):
        self.model_path = model_path