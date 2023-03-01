#include <gtest/gtest.h>
#include <fht/fht.h>

TEST(First, Small)
{
    showTime(1, 2, "test");
    EXPECT_TRUE(8 == 8);
}

TEST(First, Big)
{
    showTime(3, 4, "test1");
    EXPECT_TRUE(43181159 == 43181159);
}
