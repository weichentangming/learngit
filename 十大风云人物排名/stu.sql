/*
Navicat MySQL Data Transfer

Source Server         : xampp
Source Server Version : 100113
Source Host           : localhost:3306
Source Database       : beauty

Target Server Type    : MYSQL
Target Server Version : 100113
File Encoding         : 65001

Date: 2016-08-30 11:05:04
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for stu
-- ----------------------------
DROP TABLE IF EXISTS `stu`;
CREATE TABLE `stu` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `stu` varchar(255) DEFAULT NULL,
  `img` varchar(255) DEFAULT NULL,
  `score` int(20) NOT NULL DEFAULT '1400',
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=78 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of stu
-- ----------------------------
INSERT INTO `stu` VALUES ('1', '王尼玛1号:英俊潇洒', 'imgs/8.jpg', '1400');
INSERT INTO `stu` VALUES ('2', '王尼玛2号:风流倜傥', 'imgs/9.jpg', '1400');
INSERT INTO `stu` VALUES ('3', '王尼玛3号:玉树临风', 'imgs/10.jpg', '1400');
INSERT INTO `stu` VALUES ('4', '王尼玛4号:富甲天下', 'imgs/11.jpg', '1400');
INSERT INTO `stu` VALUES ('5', '王尼玛5号:威武雄壮', 'imgs/12.jpg', '1400');
INSERT INTO `stu` VALUES ('6', '王尼玛6号:才高八斗', 'imgs/13.jpg', '1400');
INSERT INTO `stu` VALUES ('7', '王尼玛7号:貌似潘安', 'imgs/14.jpg', '1400');
INSERT INTO `stu` VALUES ('8', '王尼玛8号:仪表堂堂', 'imgs/15.jpg', '1400');
INSERT INTO `stu` VALUES ('9', '王尼玛9号:风度翩翩', 'imgs/16.jpg', '1400');
INSERT INTO `stu` VALUES ('10', '王尼玛10号:气宇轩昂', 'imgs/17.jpg', '1400');
SET FOREIGN_KEY_CHECKS=1;
