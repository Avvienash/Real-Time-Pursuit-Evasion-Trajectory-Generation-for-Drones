import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

    
    
class PointCloudSubscriberNode(Node):
    
    def __init__(self):
        super().__init__("point_cloud_subscriber")
        self.get_logger().info("Point Cloud Subscriber Node has started")
        self.pose_subscriber = self.create_subscription(PointCloud2,"/pointCloud",self.callback,10)
        
    def callback(self, msg: PointCloud2):
        print("Received PointCloud2 message:")
        for p in pc2.read_points(msg, skip_nans=True):
            print(p)

def main(args=None):
    rclpy.init(args=args)
    
    node = PointCloudSubscriberNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
