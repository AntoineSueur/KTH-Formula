#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Float32

def talker():
    n = 4
    k = 1
    pub = rospy.Publisher('result', Float32, queue_size=10)
    rospy.init_node('publisher', anonymous=True)
    rate = rospy.Rate(0.05) # 10hz
    while not rospy.is_shutdown():
        hello_str = "K: %i" % k
        rospy.loginfo(hello_str)
        pub.publish(k)
        rate.sleep()
        k += n

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
