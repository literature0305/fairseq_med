import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Levenshtein

# leveshetein backward path (dynamic programming, diagonal first)
def lev_backward_path(seq1, seq2):
    # set score
    insertion_penalty=1
    deletion_penalty=1
    substitution_penalty=1
    go_d_score = 0

    # initialize the score matrix with zeros and size (len_y X len_x)
    len_x, len_y = len(seq1), len(seq2)
    score_matrix = np.zeros((len_y+1, len_x+1))

    # initialize path
    path_y = score_matrix.tolist()
    path_x = score_matrix.tolist()
    for j in range(len_x+1):
        for i in range(len_y+1):
            path_y[i][j] = []
            path_x[i][j] = []

    # fill the first row and first column of the score matrix
    for j in range(1, len_x+1):
        # insertion
        jj=j
        j = len_x - j
        score_matrix[-1][j] = (jj) * insertion_penalty
        path_y[-1][j].extend(path_y[-1][j+1])
        path_y[-1][j].append(len_y)
        path_x[-1][j].extend(path_x[-1][j+1])
        path_x[-1][j].append(j+1)

    for i in range(1, len_y+1):
        # deletion
        ii = i
        i = len_y - i
        score_matrix[i][-1] = ii * deletion_penalty
        path_y[i][-1].extend(path_y[i+1][-1])
        path_y[i][-1].append(i+1)
        path_x[i][-1].extend(path_x[i+1][-1])
        path_x[i][-1].append(len_x)

    # Fill the rest of the score matrix
    for i in range(1, len_y+1):
        i = len_y - i
        for j in range(1, len_x+1):
            j = len_x - j

            # update score
            if seq1[j-1] == seq2[i-1]:
                score_matrix[i][j] = score_matrix[i+1][j+1] + go_d_score
                path_y[i][j].extend(path_y[i+1][j+1])
                path_y[i][j].append(i+1)
                path_x[i][j].extend(path_x[i+1][j+1])
                path_x[i][j].append(j+1)
            else:
                go_y = (score_matrix[i+1][j] + insertion_penalty)
                go_x = (score_matrix[i][j+1] + deletion_penalty)
                go_d = (score_matrix[i+1][j+1] + substitution_penalty)
                min_avg_score = min(go_y, go_x, go_d)

                if min_avg_score == go_d:
                    score_matrix[i][j] = score_matrix[i+1][j+1] + substitution_penalty
                    path_y[i][j].extend(path_y[i+1][j+1])
                    path_y[i][j].append(i+1)
                    path_x[i][j].extend(path_x[i+1][j+1])
                    path_x[i][j].append(j+1)
                elif min_avg_score == go_y:
                    score_matrix[i][j] = score_matrix[i+1][j] + insertion_penalty
                    path_y[i][j].extend(path_y[i+1][j])
                    path_y[i][j].append(i+1)
                    path_x[i][j].extend(path_x[i+1][j])
                    path_x[i][j].append(j)
                elif min_avg_score == go_x:
                    score_matrix[i][j] = score_matrix[i][j+1] + deletion_penalty
                    path_y[i][j].extend(path_y[i][j+1])
                    path_y[i][j].append(i)
                    path_x[i][j].extend(path_x[i][j+1])
                    path_x[i][j].append(j+1)

    return score_matrix, path_x, path_y

# leveshetein forward path (dynamic programming, diagonal first)
def lev_forward_path(seq1, seq2):
    insertion_penalty=1
    deletion_penalty=1
    substitution_penalty=1

    go_d_score = 0
    go_x_score = 0
    go_y_score = 0

    # Initialize the score matrix with zeros and size len_y x len_x
    len_x, len_y = len(seq1), len(seq2)
    score_matrix = np.zeros((len_y+1, len_x+1))
    path_y = score_matrix.tolist()
    path_x = score_matrix.tolist()
    path_l = score_matrix.tolist()

    for j in range(len_x+1):
        for i in range(len_y+1):
            # initialize path & path-length
            path_y[i][j] = []
            path_x[i][j] = []
            path_l[i][j] = 1 # to avoid error (divide by 0)

    # Fill the first row and first column of the score matrix
    for i in range(1, len_x+1):
        # Insertion
        score_matrix[0][i] = i * insertion_penalty
        path_y[0][i].extend(path_y[0][i-1])
        path_y[0][i].append(0)
        path_x[0][i].extend(path_x[0][i-1])
        path_x[0][i].append(i-1)
        path_l[0][i] = i

    for j in range(1, len_y+1):
        # Deletion
        score_matrix[j][0] = j * deletion_penalty
        path_y[j][0].extend(path_y[j-1][0])
        path_y[j][0].append(j-1)
        path_x[j][0].extend(path_x[j-1][0])
        path_x[j][0].append(0)
        path_l[j][0] = j

    # Fill the rest of the score matrix
    for i in range(1, len_y+1):
        for j in range(1, len_x+1):
            # update score
            if seq1[j-1] == seq2[i-1]:
                score_matrix[i][j] = score_matrix[i-1][j-1] + go_d_score
                path_y[i][j].extend(path_y[i-1][j-1])
                path_y[i][j].append(i-1)
                path_x[i][j].extend(path_x[i-1][j-1])
                path_x[i][j].append(j-1)
            else:
                go_y = (score_matrix[i-1][j] + insertion_penalty)
                go_x = (score_matrix[i][j-1] + deletion_penalty)
                go_d = (score_matrix[i-1][j-1] + substitution_penalty)
                min_avg_score = min(go_y, go_x, go_d)

                if min_avg_score == go_d:
                    score_matrix[i][j] = score_matrix[i-1][j-1] + substitution_penalty
                    path_y[i][j].extend(path_y[i-1][j-1])
                    path_y[i][j].append(i-1)
                    path_x[i][j].extend(path_x[i-1][j-1])
                    path_x[i][j].append(j-1)
                elif min_avg_score == go_y:
                    score_matrix[i][j] = score_matrix[i-1][j] + insertion_penalty
                    path_y[i][j].extend(path_y[i-1][j])
                    path_y[i][j].append(i-1)
                    path_x[i][j].extend(path_x[i-1][j])
                    path_x[i][j].append(j)
                elif min_avg_score == go_x:
                    score_matrix[i][j] = score_matrix[i][j-1] + deletion_penalty
                    path_y[i][j].extend(path_y[i][j-1])
                    path_y[i][j].append(i)
                    path_x[i][j].extend(path_x[i][j-1])
                    path_x[i][j].append(j-1)

    return score_matrix, path_x, path_y

# leveshetein (with editdistance tool, just for check)
def lev_backward_path_fast(seq1, seq2):
    # make reverse sequence
    seq1 = '-'+seq1
    seq2 = '-'+seq2

    idx_seq1 = list(range(len(seq1)))
    idx_seq2 = list(range(len(seq2)))

    idx_seq1_after = list(range(len(seq1)))
    idx_seq2_after = list(range(len(seq2)))

    edit_log=Levenshtein.editops(seq1,seq2)

    for ele in edit_log:
        err=ele[0]
        position_a = ele[1]
        position_b = ele[2]
        if err == 'delete':
            idx_seq2_after.insert(position_b,idx_seq1[position_b])
        elif err == 'insert':
            idx_seq1_after.insert(position_a,idx_seq2[position_a])

    assert len(idx_seq1_after) == len(idx_seq2_after)
    return idx_seq1_after, idx_seq2_after

def plot_heatmap_with_optimal_path(score_matrix, seq1, seq2, path_x, path_y, name_to_write):
    # Create a heatmap using seaborn
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(score_matrix, annot=True, cmap="YlGnBu", fmt='.2f')  # Display scores with one decimal point

    # Highlight the optimal path on the heatmap
    heatmap.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', lw=2))

    for i in range(len(path_x)):
        heatmap.add_patch(plt.Rectangle((path_x[i], path_y[i]), 1, 1, fill=False, edgecolor='red', lw=2))
    
    heatmap.add_patch(plt.Rectangle((len(seq2), len(seq1)), 1, 1, fill=False, edgecolor='red', lw=2))

    # Set x-axis and y-axis labels as individual letters of the input sentences
    plt.xticks(np.arange(len(seq2)+1) + 0.5, list('-'+seq2), rotation=0)
    plt.yticks(np.arange(len(seq1)+1) + 0.5, list('-'+seq1), rotation=0)
    plt.title('Edit ditance matrix')
    plt.savefig(name_to_write)
    plt.show()

def main():
    sentence1 = 'DIVERS-'
    sentence2 = 'DRIVE-'

    # Levenshtein backward path
    score_matrix_b, path_x_b, path_y_b = lev_backward_path(sentence2, sentence1)

    # Levenshtein forward path
    score_matrix11, path_x_f, path_y_f= lev_forward_path(sentence2, sentence1)

    # Levenshtein backward path (faster)
    path_x_fast, path_y_fast = lev_backward_path_fast(sentence2, sentence1)
    for i in range(len(path_x_fast)):
        print('x:', path_x_fast[i], 'y:', path_y_fast[i])

    plt.clf() # Clear the current figure
    plot_heatmap_with_optimal_path(score_matrix_b, sentence1, sentence2, path_x_b[0][0], path_y_b[0][0], 'plot_edit_distance3-7_lbp.png')
    plt.clf() # Clear the current figure
    plot_heatmap_with_optimal_path(score_matrix11, sentence1, sentence2, path_x_f[-1][-1], path_y_f[-1][-1], 'plot_edit_distance3-7_lfp.png')
if __name__ == "__main__":
    main()
