import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scores(scores, color ):
    plt.figure(figsize=(6, 4))
    sns.histplot(scores, kde=True, bins=60, color=color)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()


def load_data(file_path="data.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_scores(data):
    data = load_data()
    post_scores = [post["score"] for post in data]
    comment_scores = [comment["score"] for post in data for comment in post["comments"]]
    total_scores = 0 
    print("Post Score Statistics:")
    num_posts = len(post_scores)
    for i in range (num_posts):
        total_scores += post_scores[i]

    mean_post_score = (total_scores / num_posts)
    postVariance = np.var(post_scores)
    standard_devation_post = np.std(post_scores)
    median_post_score = np.median(post_scores)
    print("Mean Score" , mean_post_score)
    print("Median : ",median_post_score)
    print("Variance : " , postVariance) 
    print("Standard Devation : " , standard_devation_post)
    print("\nComment Score Statistics:")
   
    total_scores_comments= 0
    num_posts_comments = len(comment_scores)
    for i in range (num_posts_comments):
        total_scores_comments += comment_scores[i]

    mean_comment_score = (total_scores_comments / num_posts_comments)
    median_comment_score = np.median(comment_scores)
    standard_devation_comment = np.std(comment_scores)
    postVariance = np.var(comment_scores)
    print("Mean comment score", mean_comment_score)
    print("Median : " , median_comment_score)
    print("Standard Devation : " ,standard_devation_comment)
    print("Variance : " , postVariance)
    
    
    
    # Plotting Score Distributions
    plot_scores(post_scores, "blue")
    plot_scores(comment_scores, "red")



def analyze_top_bottom_posts(data):
    # Add your code here. 
    # sort posts based on scores and print top 5 and bottom 5 posts.
    data = load_data()
    post_scores = [(post["score"] ,post["title"], post["created_utc"], post["subreddit"])for post in data]
    post_scores.sort()
    bottom_posts= post_scores[:6]
    top_posts = post_scores[:6]
    
    print("\nTop 5 High-Scoring Posts:")
    post_scores.reverse()
    # sorted_titles.reverse()
    top_posts= post_scores[:6]
    for i in range(6):
        print(i ,"Upvotes : ",top_posts[i], )
    



    print("\nBottom 5 Low-Scoring Posts:")
    post_scores.reverse()
    for i in range(6):
        print(i ," " ,bottom_posts[i])

    




def analyze_post_comment_correlation(data):
    
    

    data = load_data()
    post_scores = [post["score"] for post in data]
    avg_comment_scores = []
    # For every post score, compute the average of its comment scores.  
   

    for i in range (len(post_scores)):
        post = data[i]
        associated_comments = [comment["score"] for comment in post["comments"]]
        avg_comment_scores.append(sum(associated_comments)/ len(associated_comments))




    if post_scores and avg_comment_scores:
        correlation = np.corrcoef(post_scores, avg_comment_scores)[0, 1]
        print(f"\nCorrelation between post scores and average comment scores: {correlation:.2f}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=post_scores, y=avg_comment_scores)
        plt.title("Post Score vs. Average Comment Score")
        plt.xlabel("Post Score")
        plt.ylabel("Average Comment Score")
        plt.show()
    else:
        print("\nNot enough data to calculate correlation.")

if __name__ == "__main__":
    data = load_data()
    analyze_scores(data)
    analyze_top_bottom_posts(data)
    analyze_post_comment_correlation(data)

