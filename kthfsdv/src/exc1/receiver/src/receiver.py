#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32


answer = rospy.Publisher('/kthfs/results', Float32, queue_size=10)

def callback(data):
    q = 0.15
    result = data.data/q
    rospy.loginfo("Answer %f" % result )
    answer.publish(result)
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.


    rospy.init_node('receiver', anonymous=True)

    rospy.Subscriber("result", Float32, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()

