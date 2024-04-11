import sys
import pandas as pd
import os
from PyQt5.QtWidgets import *
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np  
from PyQt5.QtGui import QPixmap, QBrush, QColor
from PyQt5.QtCore import Qt

class ClusterApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Clustering')
        self.setGeometry(300, 300, 700, 500)

        image_path = 'images/bg.png'  
        pixmap = QPixmap(image_path)
        self.setPixmap(pixmap)

        button_style = """
            QPushButton {
                background-color: #876ca6;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e1cdf7;
            }
        """

        self.upload_button = QPushButton('Upload Data', self)
        self.generate_button = QPushButton('DBSCAN Clusters', self)
        self.kmeans_button = QPushButton('KMeans Clusters', self)
        self.dimensions_button = QPushButton('Select Dimensions', self)


        self.upload_button.setStyleSheet(button_style)
        self.generate_button.setStyleSheet(button_style)
        self.kmeans_button.setStyleSheet(button_style)
        self.dimensions_button.setStyleSheet(button_style)


        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.generate_button)
        buttons_layout.addWidget(self.kmeans_button)
        buttons_layout.addWidget(self.dimensions_button)



        main_layout = QVBoxLayout(self)
        main_layout.addLayout(buttons_layout)  
        main_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))  

        main_layout.setAlignment(Qt.AlignTop)

        self.upload_button.clicked.connect(self.upload_data)
        self.generate_button.clicked.connect(self.generate_clusters)
        self.kmeans_button.clicked.connect(self.generate_kmeans_clusters)
        self.dimensions_button.clicked.connect(self.choose_dimensions)



        self.data = None

    def setPixmap(self, pixmap):
        palette = self.palette()
        brush = QBrush(pixmap)
        palette.setBrush(self.backgroundRole(), brush)  
        self.setPalette(palette)

    def upload_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)", options=options)

        if file_name:
            try:
                self.data = pd.read_csv(file_name)  
            except Exception as read_error:
                print(f"Error reading file: {read_error}")

            print('Data Loaded Successfully!')

    def choose_dimensions(self):
        dimensions, ok = QInputDialog.getItem(self, "Select Dimensions", "Choose Number of Dimensions:", ["2D", "3D"], 0, False)

        if ok:
            if dimensions == "2D":
                self.generate_clusters()
            elif dimensions == "3D":
                self.generate_clusters_3d()


    def choose_columns(self):
        if self.data is not None:
            columns = self.data.columns
            x_column, ok_x = QInputDialog.getItem(self, "Select X Column", "Choose X Column:", columns, 0, False)
            y_column, ok_y = QInputDialog.getItem(self, "Select Y Column", "Choose Y Column:", columns, 0, False)

            if ok_x and ok_y:
                return x_column, y_column
            else:
                print('Column selection canceled.')
                return None, None
        else:
            print('Please upload data first.')
            return None, None

    def choose_columns_3d(self):
        if self.data is not None:
            columns = self.data.columns
            x_column, ok_x = QInputDialog.getItem(self, "Select X Column", "Choose X Column:", columns, 0, False)
            y_column, ok_y = QInputDialog.getItem(self, "Select Y Column", "Choose Y Column:", columns, 0, False)
            z_column, ok_z = QInputDialog.getItem(self, "Select Z Column (Optional)", "Choose Z Column (Optional):", columns, 0, False)

            if ok_x and ok_y:
                return x_column, y_column, z_column
            else:
                print('Column selection canceled.')
                return None, None, None
        else:
            print('Please upload data first.')
            return None, None, None


    def adjust_parameters(self, features_scaled):
        best_eps, best_min_samples, best_score = 0, 0, -1

        eps_values = np.arange(0.1, 1.0, 0.1)
        min_samples_values = range(2, 10)

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(features_scaled)

                unique_clusters = set(clusters)
                if len(unique_clusters) < 2:
                    continue

                score = silhouette_score(features_scaled, clusters)

                if score > best_score:
                    best_eps, best_min_samples, best_score = eps, min_samples, score

        return best_eps, best_min_samples

    def generate_clusters(self):
        x_column, y_column = self.choose_columns()

        if x_column and y_column:
            selected_columns = [x_column, y_column]
            features = self.data[selected_columns].values

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            best_eps, best_min_samples = self.adjust_parameters(features_scaled)

            dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
            clusters = dbscan.fit_predict(features_scaled)

            unique_clusters = set(clusters)
            if len(unique_clusters) < 2:
                print('DBSCAN generated less than 2 clusters. Please adjust parameters or choose different columns.')
                return

            colors = ['red', 'green', 'blue', 'orange']

            for i, color in zip(unique_clusters, colors):
                cluster_points = features[clusters == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {i}')

            noise_points = features[clusters == -1]
            plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', marker='x', label='Noise (Outliers)')

            plt.title('DBSCAN Clustering')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
            plt.legend()

            plt.show()
        else:
            print('Column selection canceled.')



    def generate_clusters_3d(self):
        x_column, y_column, z_column = self.choose_columns_3d()

        if x_column and y_column:
            selected_columns = [x_column, y_column]

            # Include the third column if chosen
            if z_column:
                selected_columns.append(z_column)

            features = self.data[selected_columns].values

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            best_eps, best_min_samples = self.adjust_parameters(features_scaled)

            dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
            clusters = dbscan.fit_predict(features_scaled)

            unique_clusters = set(clusters)
            if len(unique_clusters) < 2:
                print('DBSCAN generated less than 2 clusters. Please adjust parameters or choose different columns.')
                return

            # Plot 3D clusters with legend
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=clusters, cmap='viridis')

            # Create a legend
            legend_labels = [f'Cluster {cluster}' for cluster in unique_clusters]
            legend_labels.append('Noise (Outliers)')
            ax.legend(legend_labels)

            ax.set_title('DBSCAN Clustering (3D)')
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_zlabel(z_column)
            plt.show()
        else:
            print('Column selection canceled.')





    def generate_kmeans_clusters(self):
        x_column, y_column = self.choose_columns()

        if x_column and y_column:
            selected_columns = [x_column, y_column]
            features = self.data[selected_columns].values
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            kmeans = KMeans(n_clusters=3)  
            clusters = kmeans.fit_predict(features_scaled)

            plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis')
            plt.title('KMeans Clustering')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.show()
        else:
            print('Column selection canceled.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    cluster_app = ClusterApp()
    cluster_app.show()
    sys.exit(app.exec_())
