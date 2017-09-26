package edu.umass.ciir.hack.TextProcess;

import edu.umass.ciir.hack.Tools.DataProcess;
import org.json.JSONObject;
import org.lemurproject.galago.core.parse.stem.KrovetzStemmer;
import org.lemurproject.galago.tupleflow.Parameters;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;


/**
 * Created by Aqy on 11/3/16.
 */
public class AmazonReviewData_preprocess {
    public static KrovetzStemmer stemmer = new KrovetzStemmer();

    public static Reader getGzReader(String file_path) throws Exception {
        InputStream fileStream = new FileInputStream(file_path);
        InputStream gzipStream = new GZIPInputStream(fileStream);
        Reader decoder = new InputStreamReader(gzipStream, "UTF-8");
        return decoder;
    }

    public static void main(String[] args) throws Exception {

        String jsonConfigFile = args[0];//"search.params";
        String review_file = args[1];//"/Users/Aqy/Dropbox/Project/Research/ReviewEmbedding/sample_data/reviews_Musical_Instruments_5.json.gz";
        String output_file = args[2];//"/Users/Aqy/Dropbox/Project/Research/ReviewEmbedding/sample_data/reviews_Musical_Instruments_5_processed.json.gz";
        List<String> text_field_name = Arrays.asList("reviewText", "summary");

        Set<String> stopwords = null;
        if (!jsonConfigFile.toLowerCase().equals("false")){
            Parameters globalParams = Parameters.parseFile(jsonConfigFile);
            stopwords = DataProcess.getStopWords(globalParams.getAsString("stopwords"));
        }

        // Read gz file
        InputStream fileStream = new FileInputStream(review_file);
        InputStream gzipStream = new GZIPInputStream(fileStream);
        Reader decoder = new InputStreamReader(gzipStream, "UTF-8");
        BufferedReader buffered = new BufferedReader(decoder);
        Writer writer = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(output_file)), "UTF-8");
        String line = buffered.readLine();
        while( line != null){
            JSONObject obj = new JSONObject(line);
            for (String key : text_field_name){
                String input = obj.getString(key);
                List<String> terms = DataProcess.tokenize(input.replaceAll("[^a-zA-Z ]", "").toLowerCase());
                StringBuffer output = new StringBuffer();
                for (String t : terms){
                    if (stopwords != null && stopwords.contains(t)) continue;
                    output.append(stemmer.stem(t) + " ");
                }
                obj.put(key, output.toString());
            }
            //System.out.println(obj.toString());
            writer.write(obj.toString() + "\n");
            line = buffered.readLine();
        }
        writer.close();
    }
}
